#!/usr/bin/env python3
"""Back up and clear favorited ("liked") tracks, albums and/or artists from TIDAL.

Playlists are never touched: favorites and playlist membership are separate
collections in TIDAL, and this script only calls favorites.remove_* methods.
Each favorite kind is its own collection — clearing albums leaves liked tracks
and followed artists alone, and so on.

Usage:
    pip install tidalapi

    python tidal_clear_favorites.py backup --kind all
    python tidal_clear_favorites.py clear --kind artists            # dry run
    python tidal_clear_favorites.py clear --kind artists --confirm  # actually unfavorite
    python tidal_clear_favorites.py restore tidal_favorites_artists.json

Requires tidalapi >= 0.8.0 (tested against 0.8.11).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import tidalapi
    from tidalapi.exceptions import ObjectNotFound, TooManyRequests
except ImportError:
    sys.exit("tidalapi is not installed. Run: pip install tidalapi")

SESSION_FILE = Path.home() / ".tidal_session.json"
PAGE_SIZE = 100
DELETE_DELAY = 0.35  # seconds between writes; TIDAL publishes no rate limits
MAX_RETRIES = 5

log = logging.getLogger("tidal-favorites")


def _iso(value) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _track_row(track) -> dict:
    return {
        "id": track.id,
        "name": track.name,
        "artist": track.artist.name if track.artist else None,
        "album": track.album.name if track.album else None,
        "isrc": getattr(track, "isrc", None),
        "duration": track.duration,
    }


def _album_row(album) -> dict:
    return {
        "id": album.id,
        "name": album.name,
        "artist": album.artist.name if album.artist else None,
        "num_tracks": album.num_tracks,
        "upc": album.universal_product_number,
        "release_date": _iso(album.release_date),
    }


def _artist_row(artist) -> dict:
    return {
        "id": artist.id,
        "name": artist.name,
        "roles": [str(r) for r in (artist.roles or [])],
        "share_url": artist.share_url or None,
    }


# Method names on tidalapi.user.Favorites, verified against 0.8.11.
# list/count/remove are single-item for all kinds; add accepts a list.
KINDS = {
    "tracks": {
        "list": "tracks",
        "count": "get_tracks_count",
        "remove": "remove_track",
        "add": "add_track",
        "row": _track_row,
    },
    "albums": {
        "list": "albums",
        "count": "get_albums_count",
        "remove": "remove_album",
        "add": "add_album",
        "row": _album_row,
    },
    "artists": {
        "list": "artists",
        "count": "get_artists_count",
        "remove": "remove_artist",
        "add": "add_artist",
        "row": _artist_row,
    },
}

ALL_KINDS = list(KINDS)


def backup_path_for(kind: str, output: Path | None) -> Path:
    return output or Path(f"tidal_favorites_{kind}.json")


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


def fetch_favorites(session: tidalapi.Session, kind: str) -> list[dict]:
    """Page through favorites of one kind and return them as plain dicts.

    Collected in full before any deletion happens: deleting while paging
    shifts the offsets underneath you and silently skips items.
    """
    spec = KINDS[kind]
    favorites = session.user.favorites
    list_page = getattr(favorites, spec["list"])
    to_row = spec["row"]

    try:
        expected = getattr(favorites, spec["count"])()
    except Exception as exc:
        log.warning("Could not read %s count: %s", kind, exc)
        expected = None

    collected: list[dict] = []
    offset = 0
    while True:
        for attempt in range(MAX_RETRIES):
            try:
                page = list_page(limit=PAGE_SIZE, offset=offset)
                break
            except TooManyRequests:
                wait = 2**attempt
                log.warning(
                    "Rate limited fetching %s offset %s, waiting %ss",
                    kind,
                    offset,
                    wait,
                )
                time.sleep(wait)
            except Exception as exc:
                raise SystemExit(f"Failed to fetch {kind} at offset {offset}: {exc}")
        else:
            raise SystemExit(f"Gave up fetching {kind} at offset {offset}")

        if not page:
            break

        for item in page:
            try:
                collected.append(to_row(item))
            except Exception as exc:  # one malformed item shouldn't kill the export
                log.warning(
                    "Skipping unparseable %s item %s: %s",
                    kind,
                    getattr(item, "id", "?"),
                    exc,
                )

        offset += len(page)
        log.info("Fetched %s %s", offset, kind)
        if len(page) < PAGE_SIZE:
            break

    if expected is not None and len(collected) != expected:
        log.warning(
            "API reported %s %s but %s were fetched", expected, kind, len(collected)
        )

    return collected


def write_backup(items: list[dict], kind: str, path: Path) -> Path:
    """Write the backup, never clobbering an existing file."""
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = path.with_name(f"{path.stem}-{stamp}{path.suffix}")

    payload = {
        "kind": kind,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "count": len(items),
        "items": items,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s %s to %s", len(items), kind, path)
    return path


def remove_one(favorites, kind: str, item: dict) -> bool:
    """Unfavorite one item, retrying on rate limits. False means give up."""
    remove = getattr(favorites, KINDS[kind]["remove"])
    for attempt in range(MAX_RETRIES):
        try:
            remove(str(item["id"]))
            return True
        except TooManyRequests:
            wait = 2**attempt
            log.warning("Rate limited on %s, waiting %ss", item["id"], wait)
            time.sleep(wait)
        except ObjectNotFound:
            log.info("%s %s no longer in favorites, skipping", kind, item["id"])
            return True
        except Exception as exc:
            log.error(
                "Failed to remove %s %s (%s): %s",
                kind,
                item["id"],
                item.get("name"),
                exc,
            )
            return False
    return False


def clear_kind(session: tidalapi.Session, kind: str, args: argparse.Namespace) -> int:
    items = fetch_favorites(session, kind)
    if not items:
        log.info("No favorited %s found. Nothing to do.", kind)
        return 0

    backup = write_backup(items, kind, backup_path_for(kind, args.output))

    if not args.confirm:
        print(
            f"\nDry run: {len(items)} favorited {kind} would be removed.\n"
            f"Backup written to {backup}.\n"
            f"Re-run with --confirm to actually unfavorite them."
        )
        return 0

    favorites = session.user.favorites
    removed, failed = 0, []
    for i, item in enumerate(items, start=1):
        if remove_one(favorites, kind, item):
            removed += 1
        else:
            failed.append(item)
        if i % 50 == 0 or i == len(items):
            log.info("Removed %s/%s %s", removed, len(items), kind)
        time.sleep(args.delay)

    print(f"\nRemoved {removed} of {len(items)} favorited {kind}.")
    if failed:
        fail_path = backup.with_name(backup.stem + "-failed.json")
        fail_path.write_text(
            json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"{len(failed)} failed; details in {fail_path}. Re-run clear to retry them."
        )
        return 1
    print(f"Backup retained at {backup} — restore with: restore {backup}")
    return 0


def resolve_kinds(kind: str) -> list[str]:
    return ALL_KINDS if kind == "all" else [kind]


def cmd_backup(args: argparse.Namespace) -> int:
    kinds = resolve_kinds(args.kind)
    if len(kinds) > 1 and args.output:
        raise SystemExit(
            "--output cannot be combined with --kind all; run each kind separately"
        )
    session = login()
    for kind in kinds:
        items = fetch_favorites(session, kind)
        if not items:
            log.info("No favorited %s found.", kind)
            continue
        write_backup(items, kind, backup_path_for(kind, args.output))
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    kinds = resolve_kinds(args.kind)
    if len(kinds) > 1 and args.output:
        raise SystemExit(
            "--output cannot be combined with --kind all; run each kind separately"
        )
    session = login()
    return max(clear_kind(session, kind, args) for kind in kinds)


def cmd_restore(args: argparse.Namespace) -> int:
    try:
        payload = json.loads(args.backup.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read backup {args.backup}: {exc}")

    kind = args.kind or payload.get("kind")
    if kind not in KINDS:
        raise SystemExit(
            f"Backup has no usable 'kind'; pass --kind {'|'.join(ALL_KINDS)}"
        )

    items = payload.get("items", payload.get("tracks", []))
    ids = [str(i["id"]) for i in items if i.get("id")]
    if not ids:
        raise SystemExit(f"No ids found in {args.backup}")

    session = login()
    add = getattr(session.user.favorites, KINDS[kind]["add"])
    added, failed = 0, 0
    # add_* accepts a list, but batching keeps one bad id from sinking the lot.
    for i in range(0, len(ids), 20):
        chunk = ids[i : i + 20]
        try:
            add(chunk)
            added += len(chunk)
        except Exception as exc:
            failed += len(chunk)
            log.error("Failed to restore %s ids %s: %s", kind, chunk, exc)
        time.sleep(args.delay)

    print(f"Restored {added} of {len(ids)} {kind} ({failed} failed).")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--delay", type=float, default=DELETE_DELAY, help="seconds between writes"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_backup = sub.add_parser("backup", help="export favorites to JSON, change nothing")
    p_backup.add_argument("--kind", choices=ALL_KINDS + ["all"], default="tracks")
    p_backup.add_argument("-o", "--output", type=Path, default=None)
    p_backup.set_defaults(func=cmd_backup)

    p_clear = sub.add_parser(
        "clear", help="back up then unfavorite everything of a kind"
    )
    p_clear.add_argument("--kind", choices=ALL_KINDS + ["all"], default="tracks")
    p_clear.add_argument("-o", "--output", type=Path, default=None)
    p_clear.add_argument("--confirm", action="store_true", help="perform deletions")
    p_clear.set_defaults(func=cmd_clear)

    p_restore = sub.add_parser("restore", help="re-favorite items from a backup file")
    p_restore.add_argument("backup", type=Path)
    p_restore.add_argument("--kind", choices=ALL_KINDS, default=None)
    p_restore.set_defaults(func=cmd_restore)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit("\nInterrupted. Backup file (if written) is still on disk.")
