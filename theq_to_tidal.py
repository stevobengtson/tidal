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
  * The station logs back-to-back songs as one entry ("Heartbreaker/Living
    Loving Maid"); those are added as separate songs, unless TIDAL has the
    whole title as one track ("Separate Ways (Worlds Apart)").
  * Search results are cached in theq_state.json next to this script, so
    later runs only search TIDAL for songs they have not seen before. Songs
    that could not be found are retried after RETRY_MISSING_DAYS.

Usage:
    pip install tidalapi requests

    python theq_to_tidal.py --dry-run          # show what would be added
    python theq_to_tidal.py                    # add new songs to "The Q"
    python theq_to_tidal.py --days 1           # only today's plays
    python theq_to_tidal.py --playlist "Other" # different playlist name
    python theq_to_tidal.py --allow-live       # live version if no studio one

    # Why was a song not found (or matched to the wrong version)?
    python theq_to_tidal.py --explain "Black Sabbath - War Pigs"

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


_NUMBERS = {
    w: str(i)
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
# Spellings radio logs and TIDAL disagree on; mapped on both sides.
_SPELLINGS = {"tonite": "tonight", "nite": "night", "thru": "through", "n": "and"}
_NUMBER_WORD_RE = re.compile(rf"\b({'|'.join(_NUMBERS)})\b", re.I)


def norm(text: str | None) -> str:
    """Lower-case, strip accents and punctuation, drop a leading 'the'.

    Also makes "R.E.M." equal "REM", "Three" equal "3" and "Tonite" equal
    "Tonight", so the station's spelling and TIDAL's compare equal.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("&", " and ").replace("’", "'")
    # \w keeps letters from any script, so a title in Japanese or Cyrillic
    # does not collapse to an empty string.
    text = re.sub(r"[^\w ]+|_", " ", text.replace("'", ""))
    text = re.sub(r"\s+", " ", text).strip()
    # Join runs of single letters: "r e m" -> "rem", "b b king" -> "bb king".
    text = re.sub(r"\b(\w) (?=\w\b)", r"\1", text)
    words = [_SPELLINGS.get(w, _NUMBERS.get(w, w)) for w in text.split()]
    return re.sub(r"^the ", "", " ".join(words))


def words_to_digits(text: str) -> str:
    """'Three Dressed Up As A Nine' -> '3 Dressed Up As A 9' (for searching)."""
    return _NUMBER_WORD_RE.sub(lambda m: _NUMBERS[m.group(1).lower()], text)


def core_title(title: str | None) -> str:
    """Title without any bracketed or dashed suffix, normalised.

    "(Don't Fear) The Reaper" keeps "The Reaper"; that is fine because both
    sides of every comparison go through the same function.
    """
    return norm(_SUFFIX_RE.sub("", title or "")) or norm(title)


def title_variants(title: str, split_medley: bool = False) -> set[str]:
    variants = {norm(title), core_title(title)}
    if split_medley and "/" in title:
        # TIDAL titles like "War Pigs / Luke's Wall" should match "War Pigs".
        variants.add(core_title(title.split("/")[0]))
    return {v for v in variants if v}


def split_medley(title: str) -> list[str]:
    """Split a station entry like "Heartbreaker/Living Loving Maid" in two.

    The station logs songs played back to back as one entry. Parts shorter
    than three characters are not split ("AC/DC"-style titles).
    """
    parts = [p.strip() for p in re.split(r"\s*/\s*", title)]
    if len(parts) > 1 and all(len(p) >= 3 for p in parts):
        return parts
    return [title]


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


_ARTIST_JOIN_RE = re.compile(r"\b(?:and|with|feat|featuring|ft|vs|x)\b|,")


def artist_score(station_artist: str, track) -> float:
    """How well a TIDAL track's artists match the station's artist string.

    Handles the station truncating long names ("Bob Seger And The Silver
    Bullet Ba"), dropping "The" ("Beatles"), shortening names ("Sweeney Todd"
    for "The Nick Gilder Sweeney Todd Group") and joint credits ("U2 With Bb
    King" vs. a track credited to "U2" and "B.B. King").
    """
    wanted = norm(station_artist)
    wanted_parts = {p.strip() for p in _ARTIST_JOIN_RE.split(wanted) if p.strip()}
    wanted_parts = {re.sub(r"^the ", "", p) for p in wanted_parts}
    names = [a.name for a in (getattr(track, "artists", None) or []) if a.name]
    if not names and getattr(track, "artist", None):
        names = [track.artist.name]
    candidates = [norm(n) for n in names]
    candidates.append(norm(" and ".join(names)))

    best = 0.0
    for cand in filter(None, candidates):
        if cand == wanted:
            return 1.0
        if cand in wanted_parts:
            best = max(best, 0.95)  # one of several credited artists
        shorter, longer = sorted((cand, wanted), key=len)
        if len(shorter) >= 8 and longer.startswith(shorter):
            best = max(best, 0.95)  # truncated name
        if len(cand) >= 4 and re.search(rf"\b{re.escape(cand)}\b", wanted):
            best = max(best, 0.9)  # "Paul McCartney & Wings" vs "Wings"
        if len(wanted) >= 6 and re.search(rf"\b{re.escape(wanted)}\b", cand):
            best = max(best, 0.9)  # "Sweeney Todd" vs "Nick Gilder Sweeney Todd Group"
        best = max(best, similarity(cand, wanted))
    return best


def title_score(station_title: str, track) -> float:
    wanted = title_variants(station_title)
    found = title_variants(track.name or "", split_medley=True)
    best = 0.0
    for w in wanted:
        for f in found:
            best = max(best, similarity(w, f))
            # The station cuts long titles short: "It's The End Of The World"
            # for "...As We Know It (And I Feel Fine)". Compare the prefix,
            # fuzzily, so "Tonight Is A Wonderful Time" finds "Tonite Is...".
            if len(w) >= 12 and len(w.split()) >= 3 and len(f) > len(w):
                best = max(best, 0.95 * similarity(w, f[: len(w)]))
    # A title made only of symbols (e.g. "?" or "!!!") normalises to nothing.
    return best


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

    @property
    def part_keys(self) -> list[str]:
        return [
            f"{norm(self.artist)}|{core_title(p)}"
            for p in split_medley(self.search_title)
        ]

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


def primary_artist(artist: str) -> str:
    """First credited name: "U2 With Bb King" -> "U2"."""
    first = re.split(
        r"\s+(?:&|and|with|feat\.?|featuring|ft\.?)\s+", artist, flags=re.I
    )[0]
    return first.strip() or artist


def is_live_reason(reason: str) -> bool:
    return bool(re.search(r"\blive\b|unplugged|in concert", reason, re.I))


class Searcher:
    """Finds studio versions on TIDAL, caching raw search results per run."""

    def __init__(
        self,
        session: tidalapi.Session,
        delay: float,
        allow_live: bool = False,
        explain: bool = False,
    ):
        self.session = session
        self.delay = delay
        self.allow_live = allow_live
        self.explain = explain
        self.calls = 0
        self._results: dict[tuple[str, int], list] = {}

    def _search(self, query: str, limit: int) -> list:
        key = (query.lower(), limit)
        if key not in self._results:
            result = with_retry(
                f"searching '{query}'",
                self.session.search,
                query,
                models=[tidalapi.Track],
                limit=limit,
            )
            self.calls += 1
            time.sleep(self.delay)
            self._results[key] = list(result.get("tracks") or [])
            if self.explain:
                print(f"  search '{query}': {len(self._results[key])} results")
        return self._results[key]

    @staticmethod
    def _queries(artist: str, title: str) -> list[tuple[str, int]]:
        flat = title.replace("/", " ")
        core = _SUFFIX_RE.sub("", title).strip().replace("/", " ") or flat
        primary = primary_artist(artist)
        candidates = [
            f"{artist} {flat}",
            f"{artist} {core}",
            f"{primary} {flat}",  # truncated/joint artist credits
            f"{primary} {words_to_digits(core)}",  # "Three" -> "3"
            flat,  # odd artist spellings
        ]
        queries, seen = [], set()
        for q in candidates:
            q = re.sub(r"\s+", " ", q).strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                queries.append((q, SEARCH_LIMIT))
        # Last resort: the artist's top tracks, scored fuzzily. Catches
        # truncated or misspelt titles TIDAL's search does not.
        queries.append((primary, 50))
        return queries

    def find(
        self,
        artist: str,
        title: str,
        min_title: float = MIN_TITLE_SCORE,
        allow_live: bool = False,
    ):
        """Best studio-version match for one song, or None.

        With allow_live, a live version is returned when there is no studio one.
        """
        seen: set = set()
        best, best_rank = None, float("-inf")
        live, live_rank = None, float("-inf")

        for query, limit in self._queries(artist, title):
            for position, track in enumerate(self._search(query, limit)):
                if track.id in seen:
                    continue
                seen.add(track.id)

                t_score = title_score(title, track)
                a_score = artist_score(artist, track)
                if t_score < min_title or a_score < MIN_ARTIST_SCORE:
                    if self.explain and t_score >= 0.6 and a_score >= 0.5:
                        print(
                            f"    no  t={t_score:.2f} a={a_score:.2f} "
                            f"{describe(track)}: title/artist too different"
                        )
                    continue

                album = getattr(track, "album", None)
                album_name = getattr(album, "name", "") or ""
                album_type = (getattr(album, "type", "") or "").upper()
                rank = 2 * t_score + a_score - 0.01 * position
                if album_type == "COMPILATION" or COMPILATION_RE.search(album_name):
                    rank -= 0.3
                if album_type in ("SINGLE", "EP"):
                    rank -= 0.05

                reason = unwanted_reason(track, title)
                if reason is None and getattr(track, "available", True) is False:
                    reason = "not streamable"
                if self.explain:
                    verdict = f"no  ({reason})" if reason else f"ok  rank={rank:.2f}"
                    print(
                        f"    {verdict} t={t_score:.2f} a={a_score:.2f} "
                        f"{describe(track)}"
                    )
                if reason:
                    log.debug("  reject %s: %s", describe(track), reason)
                    if is_live_reason(reason) and rank > live_rank:
                        live, live_rank = track, rank
                    continue
                if rank > best_rank:
                    best, best_rank = track, rank

            if best is not None and best_rank >= 2.9:
                break  # near-perfect hit; no need for fallback queries

        if best is None and allow_live and live is not None:
            log.info("No studio version of %s - %s; using live", artist, title)
            return live
        return best

    def resolve(self, play: Play) -> tuple[list, list[str]]:
        """Return (tracks to add, titles not found) for one station entry.

        Studio versions are always tried first; live versions (with
        --allow-live) only fill in what is still missing at the end.
        """
        parts = split_medley(play.search_title)
        if len(parts) == 1:
            track = self.find(play.artist, play.search_title) or (
                self.allow_live
                and self.find(play.artist, play.search_title, allow_live=True)
            )
            return ([track], []) if track else ([], [play.search_title])

        # "Separate Ways/Worlds Apart" and "867-5309/Jenny" are one song;
        # only take the whole title when TIDAL has it nearly verbatim.
        whole = self.find(play.artist, play.search_title, min_title=0.95)
        if whole is not None:
            return [whole], []

        found, missing = [], []
        for part in parts:
            track = self.find(play.artist, part)
            if track is None:
                missing.append(part)
            elif all(t.id != track.id for t in found):
                found.append(track)

        if missing and self.allow_live:
            if not found:
                whole = self.find(
                    play.artist, play.search_title, min_title=0.95, allow_live=True
                )
                if whole is not None:
                    return [whole], []
            still_missing = []
            for part in missing:
                track = self.find(play.artist, part, allow_live=True)
                if track is None:
                    still_missing.append(part)
                elif all(t.id != track.id for t in found):
                    found.append(track)
            missing = still_missing
        return found, missing


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


STATE_VERSION = 2


def load_state(path: Path) -> dict:
    fresh = {"version": STATE_VERSION, "matches": {}}
    if not path.exists():
        return fresh
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read state file {path}: {exc}")
    if state.get("version") != STATE_VERSION:
        # Matching rules changed; old cached results would hide the fixes.
        log.info("Matching rules changed since the last run; clearing the cache")
        return fresh
    state.setdefault("matches", {})
    return state


def save_state(path: Path, state: dict) -> None:
    state["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def cached_entry(state: dict, play: Play, retry_missing: bool) -> dict | None:
    """The cached result for a play, or None if it should be searched again."""
    entry = state["matches"].get(play.key)
    if entry is None:
        return None
    if not entry.get("missing"):
        return entry
    if retry_missing:
        return None
    try:
        checked = datetime.fromisoformat(entry["checked"])
    except (KeyError, ValueError):
        return None
    if datetime.now(timezone.utc) - checked > timedelta(days=RETRY_MISSING_DAYS):
        return None
    return entry


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def explain(args: argparse.Namespace) -> int:
    """Print every TIDAL candidate for the given songs and why it was chosen."""
    session = login()
    searcher = Searcher(session, args.delay, allow_live=args.allow_live, explain=True)
    for spec in args.explain:
        artist, sep, title = spec.partition(" - ")
        if not sep:
            raise SystemExit(f"--explain needs 'Artist - Title', got {spec!r}")
        play = Play(artist.strip(), title.strip(), strip_unwanted_markers(title.strip()), 0)
        print(f"\n{play}  (searching for '{play.search_title}')")
        tracks, missing = searcher.resolve(play)
        for track in tracks:
            print(f"  => {describe(track)}")
        for part in missing:
            print(f"  => not found: {part}")
    return 0


def run(args: argparse.Namespace) -> int:
    if args.explain:
        return explain(args)

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

    searcher = Searcher(session, args.delay, allow_live=args.allow_live)
    to_add: list[tuple[Play, str, str]] = []
    unmatched: list[str] = []
    skipped = searched = 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for play in plays:
        if play.key in have_keys or all(k in have_keys for k in play.part_keys):
            skipped += 1
            continue

        entry = cached_entry(state, play, args.retry_missing)
        if entry is None:
            searched += 1
            tracks, missing = searcher.resolve(play)
            entry = {
                "artist": play.artist,
                "title": play.title,
                "tracks": [
                    {"id": str(t.id), "isrc": t.isrc, "match": describe(t)}
                    for t in tracks
                ],
                "missing": missing,
                "checked": now,
            }
            state["matches"][play.key] = entry
            if searched % 25 == 0:
                save_state(args.state, state)  # keep progress if interrupted

        new_here = 0
        for track in entry["tracks"]:
            tidal_id, isrc = track["id"], track.get("isrc")
            if tidal_id in have_ids or (isrc and isrc in have_isrcs):
                continue
            to_add.append((play, tidal_id, track.get("match") or tidal_id))
            have_ids.add(tidal_id)
            if isrc:
                have_isrcs.add(isrc)
            new_here += 1
        if entry["tracks"] and not new_here:
            skipped += 1
        have_keys.add(play.key)
        unmatched += [f"{play.artist} - {part}" for part in entry.get("missing", [])]

    save_state(args.state, state)

    print(
        f"\n{len(plays)} songs from the station: {skipped} already in playlist, "
        f"{len(to_add)} new tracks, {len(unmatched)} not found on TIDAL "
        f"({searched} songs searched, {searcher.calls} TIDAL searches this run)."
    )
    for play, _, label in to_add:
        print(f"  + {play}  ->  {label}")
    if unmatched:
        print("\nNot found on TIDAL (studio version):")
        for line in unmatched:
            print(f"  ? {line}")
        print("Check one with: python theq_to_tidal.py --explain 'Artist - Title'")

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
            with_retry("adding tracks", playlist.add, chunk, allow_duplicates=False)
            added += len(chunk)
        except Exception as exc:
            failed += len(chunk)
            log.error("Failed to add %d tracks: %s", len(chunk), exc)
        time.sleep(args.delay)

    summary = f"\nAdded {added} tracks to '{playlist.name}'"
    print(summary + (f", {failed} failed." if failed else "."))
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
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="use a live version when TIDAL has no studio version of a song",
    )
    parser.add_argument(
        "--explain",
        action="append",
        metavar="'ARTIST - TITLE'",
        help="show every TIDAL candidate for a song and why it was or was not "
        "chosen, then exit (repeatable; changes nothing)",
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
