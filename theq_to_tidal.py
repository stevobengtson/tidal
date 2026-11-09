#!/usr/bin/env python3
"""Sync songs played on 100.3 The Q! (theq.fm) into a TIDAL playlist.

Reads the station's broadcast history (the same JSON feed that powers
https://www.theq.fm/playlist/, covering the last 7 days), finds the studio
version of each song on TIDAL and appends any that are not already in the
playlist. Safe to run repeatedly, e.g. from cron:

  * Songs already in the playlist are skipped, matched by TIDAL id, ISRC and
    by normalised artist + title, so a different release of the same song
    (album vs. single vs. remaster) is not added twice.
  * Live, unplugged, acoustic, demo, remix, karaoke and cover versions are
    rejected. When the station itself lists a song as "(Live)", the studio
    original is looked up instead.
  * Search results are cached in theq_state.json next to this script, so
    later runs only search TIDAL for songs they have not seen before. Songs
    that could not be found are retried after RETRY_MISSING_DAYS.

Usage:
    pip install tidalapi requests

    python theq_to_tidal.py --dry-run          # show what would be added
    python theq_to_tidal.py                    # add new songs to "The Q"
    python theq_to_tidal.py --days 1           # only today's plays
    python theq_to_tidal.py --playlist "Other" # different playlist name

Shares the login token file (~/.tidal_session.json) with
tidal_clear_favourites.py. Requires tidalapi >= 0.8.0 (tested against 0.8.11).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests is not installed. Run: pip install requests")

try:
    import tidalapi
    from tidalapi.exceptions import TooManyRequests
except ImportError:
    sys.exit("tidalapi is not installed. Run: pip install tidalapi")

SESSION_FILE = Path.home() / ".tidal_session.json"
STATE_FILE = Path(__file__).resolve().parent / "theq_state.json"

STATION_API = "https://www.theq.fm/api/v1/music/broadcastHistory"
STATION_PLAYER_ID = "862"  # data-player on the theq.fm playlist widget
STATION_TZ = "America/Vancouver"
MAX_DAYS = 7  # the station only exposes today plus the previous 6 days

DEFAULT_PLAYLIST = "The Q"
SEARCH_LIMIT = 25
ADD_CHUNK = 50
DEFAULT_DELAY = 0.35  # seconds between TIDAL calls; TIDAL publishes no limits
MAX_RETRIES = 5
RETRY_MISSING_DAYS = 14

# Minimum similarity (0-1) for a TIDAL result to count as the same song.
MIN_TITLE_SCORE = 0.85
MIN_ARTIST_SCORE = 0.80

log = logging.getLogger("theq-to-tidal")


# --------------------------------------------------------------------------
# Text normalisation and version detection
# --------------------------------------------------------------------------

# Words that mark a recording as something other than the studio original.
_UNWANTED = (
    r"live|unplugged|acoustic|demo|remix|re-?mix|karaoke|instrumental|cover|"
    r"tribute|re-?recorded|rehearsal|session|in concert|commentary|"
    r"sped up|slowed|lullaby|8-bit|piano version|orchestral|symphonic|"
    r"club mix|dub|extended mix|a cappella|acapella"
)
UNWANTED_RE = re.compile(rf"\b(?:{_UNWANTED})\b", re.I)

# Albums whose name says they are live recordings or tribute/karaoke sets.
LIVE_ALBUM_RE = re.compile(
    r"\blive\b\s*(?:at|in|from|on|!|$)|^live\b|\(live\)|\[live\]|unplugged|"
    r"in concert|mtv|bbc sessions|karaoke|tribute|made famous|"
    r"as performed by|in the style of",
    re.I,
)

# Compilations are acceptable but a studio album or single is preferred.
COMPILATION_RE = re.compile(
    r"greatest hits|best of|\bhits\b|collection|anthology|essential|"
    r"\bgold\b|definitive|ultimate|decade|singles|now that'?s|various",
    re.I,
)

# Bracketed or " - " suffixes, e.g. "Song (Live)", "Song - 2011 Remaster".
_SUFFIX_RE = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]|\s+-\s+.*$")


def norm(text: str | None) -> str:
    """Lower-case, strip accents and punctuation, drop a leading 'the'."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ").replace("’", "'")
    text = re.sub(r"[^a-z0-9 ]+", " ", text.replace("'", ""))
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^the ", "", text)


def core_title(title: str | None) -> str:
    """Title without any bracketed or dashed suffix, normalised.

    "(Don't Fear) The Reaper" keeps "The Reaper"; that is fine because both
    sides of every comparison go through the same function.
    """
    return norm(_SUFFIX_RE.sub("", title or "")) or norm(title)


def title_variants(title: str) -> set[str]:
    return {v for v in (norm(title), core_title(title)) if v}


def strip_unwanted_markers(title: str) -> str:
    """Remove "(Live)", "[Unplugged]", " - Live at X" etc. from a station title.

    Parentheticals that are part of the real title, like
    "(Don't Fear) The Reaper", are kept.
    """

    def drop(match: re.Match) -> str:
        return "" if UNWANTED_RE.search(match.group(0)) else match.group(0)

    cleaned = re.sub(r"\s*[\(\[][^\)\]]*[\)\]]", drop, title)
    cleaned = re.sub(r"\s+-\s+.*$", drop, cleaned)
    return cleaned.strip() or title.strip()


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def artist_score(station_artist: str, track) -> float:
    """How well a TIDAL track's artists match the station's artist string.

    Handles the station truncating long names ("Nathaniel Rateliff & The
    Night Swe"), dropping "The" ("Beatles") and joint credits ("Paul McCartney
    & Wings" vs. a track credited to "Wings").
    """
    wanted = norm(station_artist)
    names = [a.name for a in (getattr(track, "artists", None) or []) if a.name]
    if not names and getattr(track, "artist", None):
        names = [track.artist.name]
    candidates = [norm(n) for n in names]
    candidates.append(norm(" and ".join(names)))

    best = 0.0
    for cand in filter(None, candidates):
        if cand == wanted:
            return 1.0
        shorter, longer = sorted((cand, wanted), key=len)
        if len(shorter) >= 8 and longer.startswith(shorter):
            best = max(best, 0.95)  # truncated name
        elif len(cand) >= 4 and re.search(rf"\b{re.escape(cand)}\b", wanted):
            best = max(best, 0.9)  # one of several credited artists
        best = max(best, similarity(cand, wanted))
    return best


def title_score(station_title: str, track) -> float:
    wanted = title_variants(station_title)
    found = title_variants(track.name or "")
    return max(similarity(w, f) for w in wanted for f in found)


def unwanted_reason(track, station_title: str) -> str | None:
    """Why a TIDAL track is not a studio original, or None if it is fine."""
    version = getattr(track, "version", None) or ""
    if UNWANTED_RE.search(version):
        return f"version '{version}'"

    # Only look at the title's suffix: "Live And Let Die" or "Alive" are fine.
    suffix = (track.name or "")[len(_SUFFIX_RE.sub("", track.name or "")) :]
    if UNWANTED_RE.search(suffix):
        return f"title '{track.name}'"

    album = getattr(track, "album", None)
    album_name = getattr(album, "name", "") or ""
    # A song called "Live Forever" on an album also called that is fine.
    if LIVE_ALBUM_RE.search(album_name) and core_title(station_title) not in norm(
        album_name
    ):
        return f"album '{album_name}'"
    return None


# --------------------------------------------------------------------------
# Station feed
# --------------------------------------------------------------------------


@dataclass
class Play:
    artist: str
    title: str  # as the station lists it
    search_title: str  # with "(Live)" etc. removed
    played_at: int  # unix timestamp of the earliest play seen

    @property
    def key(self) -> str:
        return f"{norm(self.artist)}|{core_title(self.search_title)}"

    def __str__(self) -> str:
        return f"{self.artist} - {self.title}"


def fetch_station_plays(days: int) -> list[Play]:
    """Return unique songs played over the last `days` days, oldest first."""
    http = requests.Session()
    http.headers["User-Agent"] = "Mozilla/5.0 (playlist sync script)"
    plays: dict[str, Play] = {}

    for day in range(0, -days, -1):
        params = {"day": day, "playerID": STATION_PLAYER_ID, "tz": STATION_TZ}
        try:
            resp = http.get(STATION_API, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.warning("Could not fetch station history for day %s: %s", day, exc)
            continue

        if payload.get("status") != "success":
            log.warning("Station returned %r for day %s", payload.get("status"), day)
            continue

        data = payload.get("data") or {}
        songs = data.get("songs") or []
        log.info("%s: %d plays", data.get("date", f"day {day}"), len(songs))

        for song in songs:
            artist = (song.get("artist_name") or "").strip()
            title = (song.get("song_name") or "").strip()
            if not artist or not title:
                continue
            play = Play(
                artist=artist,
                title=title,
                search_title=strip_unwanted_markers(title),
                played_at=int(song.get("last_played") or 0),
            )
            existing = plays.get(play.key)
            if existing is None or play.played_at < existing.played_at:
                plays[play.key] = play

    return sorted(plays.values(), key=lambda p: p.played_at)


# --------------------------------------------------------------------------
# TIDAL
# --------------------------------------------------------------------------


def with_retry(what: str, func, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except TooManyRequests:
            wait = 2 ** (attempt + 1)
            log.warning("Rate limited while %s, waiting %ss", what, wait)
            time.sleep(wait)
    raise SystemExit(f"Gave up {what} after {MAX_RETRIES} rate-limited attempts")


def login() -> tidalapi.Session:
    """Log in, reusing a cached token file when one is present."""
    session = tidalapi.Session()
    try:
        session.login_session_file(SESSION_FILE, do_pkce=False)
    except Exception as exc:  # network failure, corrupt token file, etc.
        raise SystemExit(f"Login failed: {exc}") from exc

    if not session.check_login():
        raise SystemExit(f"Login did not complete. Delete {SESSION_FILE} and retry.")

    log.info("Logged in as user %s", session.user.id)
    return session


def find_playlist(
    session: tidalapi.Session, name: str, playlist_id: str | None, create: bool
):
    if playlist_id:
        playlist = session.playlist(playlist_id)
        if not isinstance(playlist, tidalapi.UserPlaylist):
            playlist = playlist.factory()
        if not isinstance(playlist, tidalapi.UserPlaylist):
            raise SystemExit(f"Playlist {playlist_id} is not one of your own playlists")
        return playlist

    matches = []
    offset = 0
    while True:
        page = with_retry(
            "listing playlists",
            session.user.playlist_and_favorite_playlists,
            offset=offset,
            limit=50,
        )
        if not page:
            break
        matches += [
            p
            for p in page
            if isinstance(p, tidalapi.UserPlaylist)
            and (p.name or "").strip().lower() == name.strip().lower()
        ]
        if len(page) < 50:
            break
        offset += 50

    if len(matches) > 1:
        ids = ", ".join(p.id for p in matches)
        raise SystemExit(
            f"Found {len(matches)} playlists named '{name}' ({ids}); "
            "pick one with --playlist-id"
        )
    if matches:
        return matches[0]
    if not create:
        raise SystemExit(f"No playlist named '{name}' found (use --create to make it)")

    log.info("Creating playlist '%s'", name)
    return session.user.create_playlist(
        name, "Songs played on 100.3 The Q! — synced from theq.fm/playlist"
    )


def playlist_tracks(playlist) -> list:
    """All tracks currently in the playlist."""
    tracks: list = []
    offset = 0
    while True:
        page = with_retry(
            "reading playlist", playlist.tracks, limit=100, offset=offset
        )
        if not page:
            break
        tracks += page
        offset += len(page)
    return tracks


def track_key(track) -> str:
    artist = track.artist.name if getattr(track, "artist", None) else ""
    return f"{norm(artist)}|{core_title(track.name)}"


def describe(track) -> str:
    artist = track.artist.name if getattr(track, "artist", None) else "?"
    album = track.album.name if getattr(track, "album", None) else "?"
    name = getattr(track, "full_name", None) or track.name
    return f"{artist} - {name} [{album}]"


def search_tidal(session: tidalapi.Session, play: Play, delay: float):
    """Find the best studio-version match for a station play, or None."""
    queries = [f"{play.artist} {play.search_title}"]
    core = _SUFFIX_RE.sub("", play.search_title).strip()
    if core and core != play.search_title:
        queries.append(f"{play.artist} {core}")
    # Truncated or odd artist credits sometimes sink the combined query.
    queries.append(play.search_title)

    seen: set[int] = set()
    best, best_rank = None, float("-inf")

    for query in queries:
        result = with_retry(
            f"searching '{query}'",
            session.search,
            query,
            models=[tidalapi.Track],
            limit=SEARCH_LIMIT,
        )
        time.sleep(delay)

        for position, track in enumerate(result.get("tracks") or []):
            if track.id in seen:
                continue
            seen.add(track.id)

            t_score = title_score(play.search_title, track)
            a_score = artist_score(play.artist, track)
            if t_score < MIN_TITLE_SCORE or a_score < MIN_ARTIST_SCORE:
                continue
            reason = unwanted_reason(track, play.search_title)
            if reason:
                log.debug("  reject %s: %s", describe(track), reason)
                continue
            if getattr(track, "available", True) is False:
                continue

            album = getattr(track, "album", None)
            album_name = getattr(album, "name", "") or ""
            album_type = (getattr(album, "type", "") or "").upper()

            rank = 2 * t_score + a_score - 0.01 * position
            if album_type == "COMPILATION" or COMPILATION_RE.search(album_name):
                rank -= 0.3
            if album_type in ("SINGLE", "EP"):
                rank -= 0.05
            if rank > best_rank:
                best, best_rank = track, rank

        if best is not None and best_rank >= 2.9:
            break  # near-perfect hit; no need for fallback queries

    return best


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"matches": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read state file {path}: {exc}")
    state.setdefault("matches", {})
    return state


def save_state(path: Path, state: dict) -> None:
    state["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def cached_match(state: dict, play: Play, retry_missing: bool):
    """Return (hit, tidal_id) from the cache. hit=False means search again."""
    entry = state["matches"].get(play.key)
    if entry is None:
        return False, None
    if entry.get("tidal_id"):
        return True, entry["tidal_id"]
    if retry_missing:
        return False, None
    try:
        checked = datetime.fromisoformat(entry["checked"])
    except (KeyError, ValueError):
        return False, None
    if datetime.now(timezone.utc) - checked > timedelta(days=RETRY_MISSING_DAYS):
        return False, None
    return True, None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    plays = fetch_station_plays(args.days)
    if not plays:
        print("No songs found in the station history.")
        return 1
    log.info("%d unique songs played in the last %d day(s)", len(plays), args.days)

    state = load_state(args.state)
    session = login()
    playlist = find_playlist(
        session, args.playlist, args.playlist_id, create=args.create
    )
    existing = playlist_tracks(playlist)
    log.info("Playlist '%s' has %d tracks", playlist.name, len(existing))

    have_ids = {str(t.id) for t in existing}
    have_isrcs = {t.isrc for t in existing if getattr(t, "isrc", None)}
    have_keys = {track_key(t) for t in existing}

    to_add: list[tuple[Play, str, str]] = []
    unmatched: list[Play] = []
    skipped = searched = 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for play in plays:
        if play.key in have_keys:
            skipped += 1
            continue

        hit, tidal_id = cached_match(state, play, args.retry_missing)
        label = state["matches"].get(play.key, {}).get("match")
        isrc = state["matches"].get(play.key, {}).get("isrc")
        if not hit:
            searched += 1
            track = search_tidal(session, play, args.delay)
            if track is None:
                state["matches"][play.key] = {
                    "artist": play.artist,
                    "title": play.title,
                    "tidal_id": None,
                    "checked": now,
                }
            else:
                tidal_id, isrc, label = str(track.id), track.isrc, describe(track)
                state["matches"][play.key] = {
                    "artist": play.artist,
                    "title": play.title,
                    "tidal_id": tidal_id,
                    "isrc": isrc,
                    "match": label,
                    "checked": now,
                }
            if searched % 25 == 0:
                save_state(args.state, state)  # keep progress if interrupted

        if not tidal_id:
            unmatched.append(play)
            continue
        if tidal_id in have_ids or (isrc and isrc in have_isrcs):
            skipped += 1
            have_keys.add(play.key)
            continue

        to_add.append((play, tidal_id, label or tidal_id))
        have_ids.add(tidal_id)
        have_keys.add(play.key)
        if isrc:
            have_isrcs.add(isrc)

    save_state(args.state, state)

    print(
        f"\n{len(plays)} songs from the station: {skipped} already in playlist, "
        f"{len(to_add)} new, {len(unmatched)} not found on TIDAL "
        f"({searched} TIDAL searches this run)."
    )
    for play, _, label in to_add:
        print(f"  + {play}  ->  {label}")
    if unmatched:
        print("\nNot found on TIDAL (studio version):")
        for play in unmatched:
            print(f"  ? {play}")

    if not to_add:
        return 0
    if args.dry_run:
        print("\nDry run: nothing was added. Re-run without --dry-run to add.")
        return 0

    added = failed = 0
    ids = [tidal_id for _, tidal_id, _ in to_add]
    for i in range(0, len(ids), ADD_CHUNK):
        chunk = ids[i : i + ADD_CHUNK]
        try:
            with_retry(
                "adding tracks", playlist.add, chunk, allow_duplicates=False
            )
            added += len(chunk)
        except Exception as exc:
            failed += len(chunk)
            log.error("Failed to add %d tracks: %s", len(chunk), exc)
        time.sleep(args.delay)

    print(f"\nAdded {added} tracks to '{playlist.name}'" + (f", {failed} failed" if failed else "") + ".")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--playlist", default=DEFAULT_PLAYLIST, help="playlist name (default: %(default)s)"
    )
    parser.add_argument("--playlist-id", help="playlist UUID; overrides --playlist")
    parser.add_argument(
        "--create", action="store_true", help="create the playlist if it is missing"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=MAX_DAYS,
        choices=range(1, MAX_DAYS + 1),
        metavar=f"1-{MAX_DAYS}",
        help="days of station history to read (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would be added, change nothing"
    )
    parser.add_argument(
        "--retry-missing",
        action="store_true",
        help="search again for songs not found on earlier runs",
    )
    parser.add_argument(
        "--state", type=Path, default=STATE_FILE, help="cache file (default: %(default)s)"
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY, help="seconds between TIDAL calls"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    # tidalapi/urllib3 are chatty at DEBUG; keep -v about this script.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("tidalapi").setLevel(logging.WARNING)
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Re-run to pick up where it left off.")
