from __future__ import annotations

import json
import re
import shutil
import sqlite3
import tempfile
import urllib.parse
import warnings
from collections.abc import Iterator
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from zotero_cli_cc.core.fts import CONTENT_INDEX_VERSION, content_match_clause
from zotero_cli_cc.models import (
    Attachment,
    Collection,
    Creator,
    DuplicateGroup,
    Item,
    Note,
    OrphanAttachment,
    SearchResult,
)

# Zotero file sync states that mean "the server still has a copy to pull down".
_SYNC_STATE_TO_DOWNLOAD = 1
_SYNC_STATE_FORCE_DOWNLOAD = 4

# Excluded type names (looked up dynamically per database)
_EXCLUDED_TYPE_NAMES = ("attachment", "note", "annotation")

# Tested schema version range (Zotero 6–10; Zotero 10 is userdata schema 129)
MIN_SCHEMA_VERSION = 100
MAX_SCHEMA_VERSION = 200

# Busy timeout for the direct read-only open. Zotero 10 holds
# PRAGMA locking_mode=EXCLUSIVE on zotero.sqlite for the app's whole lifetime,
# so a locked database never clears — fail fast and fall back to a snapshot.
_CONNECT_BUSY_TIMEOUT_S = 1.5

# Busy timeout for fulltext.sqlite reads. Zotero 10 does not lock this
# database exclusively, but its indexer writes in bursts — wait those out.
_FULLTEXT_BUSY_TIMEOUT_S = 5.0

# SQLITE_MAX_VARIABLE_NUMBER can be as low as 999 on older SQLite builds, so
# large IN (...) parameter lists are executed in batches. Sorted queries keep a
# single SQL ORDER BY up to _SORT_IN_THRESHOLD ids (modern SQLite allows 32766
# variables) and only fall back to Python-side sorting beyond that.
_IN_BATCH_SIZE = 900
_SORT_IN_THRESHOLD = 30000

_NOCASE_TRANS = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def _in_batches(ids: list[int], size: int | None = None) -> Iterator[list[int]]:
    """Yield an id list in chunks small enough for SQLite's variable limit."""
    if size is None:
        size = _IN_BATCH_SIZE  # read at call time so tests can monkeypatch
    for i in range(0, len(ids), size):
        yield ids[i : i + size]


def _nocase_key(value: str) -> str:
    """ASCII-only case fold, matching SQLite's COLLATE NOCASE."""
    return value.translate(_NOCASE_TRANS)


class ZoteroReader:
    """Read-only access to the local Zotero SQLite database.

    Connection semantics depend on whether the Zotero desktop app is running:

    - Zotero closed: reads go straight to zotero.sqlite (read-only, WAL-aware)
      and are always live.
    - Zotero 10 running: the app holds PRAGMA locking_mode=EXCLUSIVE on
      zotero.sqlite for its whole lifetime, so no external process can open it.
      The Reader then works on a consistent snapshot — a copy of zotero.sqlite
      plus its WAL taken when the Reader first connects — and reuses that
      snapshot until close(). Writes Zotero makes afterwards are not visible;
      close and reopen the Reader to refresh.
    """

    def __init__(
        self,
        db_path: Path,
        library_id: int = 1,
        prefs_js_path: Path | None = None,
        fulltext_db_path: Path | None = None,
    ) -> None:
        self._db_path = db_path
        self._library_id = library_id
        # Zotero 10's FTS5 full-text index lives next to zotero.sqlite.
        self._fulltext_db_path = fulltext_db_path or db_path.with_name("fulltext.sqlite")
        self._conn: sqlite3.Connection | None = None
        self._ft_conn: sqlite3.Connection | None = None
        self._ft_checked = False
        self._tmp_dir: Path | None = None
        self._excluded_sql: str | None = None
        self._excluded_ids: tuple[int, ...] | None = None
        self._tmp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
        from zotero_cli_cc.core.attachment_resolver import AttachmentResolver

        self._resolver = AttachmentResolver(db_path, prefs_js_path)

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        if not self._db_path.exists():
            raise FileNotFoundError(
                f"Zotero database not found: {self._db_path}\n"
                f"  Run 'zot config show' to check your configuration.\n"
                f"  Run 'zot config init --data-dir <path>' to set the correct data directory."
            )
        # Direct read-only open: live, WAL-aware data (a plain mode=ro open
        # reads through the WAL). Percent-encode so paths containing '?', '#'
        # or '%' don't corrupt the URI.
        uri_path = urllib.parse.quote(self._db_path.as_posix())
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(
                f"file:{uri_path}?mode=ro",
                uri=True,
                timeout=_CONNECT_BUSY_TIMEOUT_S,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("SELECT 1 FROM items LIMIT 1")
            self._conn = conn
            return conn
        except sqlite3.OperationalError:
            if conn is not None:
                conn.close()
            # Locked (Zotero 10 holds an exclusive lock while running) or
            # otherwise unreadable — fall back to a snapshot copy of DB + WAL.
            return self._connect_from_snapshot()

    def _connect_fulltext(self) -> sqlite3.Connection | None:
        """Open Zotero 10's FTS5 full-text index (fulltext.sqlite), if present.

        Unlike zotero.sqlite, fulltext.sqlite is not exclusively locked while
        Zotero 10 runs, so a plain read-only open suffices; the busy timeout
        rides out the indexer reindexing write bursts. Returns None when the
        file is absent (pre-Zotero 10 data directories) or unreadable. The
        result (including a miss) is cached until close().
        """
        if self._ft_checked:
            return self._ft_conn
        self._ft_checked = True
        if not self._fulltext_db_path.exists():
            return None
        uri_path = urllib.parse.quote(self._fulltext_db_path.as_posix())
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(
                f"file:{uri_path}?mode=ro",
                uri=True,
                timeout=_FULLTEXT_BUSY_TIMEOUT_S,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("SELECT 1 FROM fulltextIndexState LIMIT 1")
        except sqlite3.OperationalError:
            if conn is not None:
                conn.close()
            return None
        self._ft_conn = conn
        return conn

    def _get_excluded_ids(self) -> tuple[int, ...]:
        """Look up excluded type IDs by name (cached after first call)."""
        if self._excluded_ids is not None:
            return self._excluded_ids
        conn = self._connect()
        placeholders = ",".join("?" * len(_EXCLUDED_TYPE_NAMES))
        rows = conn.execute(
            f"SELECT itemTypeID FROM itemTypes WHERE typeName IN ({placeholders})",
            _EXCLUDED_TYPE_NAMES,
        ).fetchall()
        self._excluded_ids = tuple(r["itemTypeID"] for r in rows) if rows else (-1,)
        return self._excluded_ids

    def _get_excluded_sql(self) -> str:
        """Build SQL fragment with literal IDs (for simple string concatenation)."""
        if self._excluded_sql is not None:
            return self._excluded_sql
        ids = self._get_excluded_ids()
        self._excluded_sql = f"NOT IN ({','.join(str(i) for i in ids)})"
        return self._excluded_sql

    def _excluded_filter(self) -> tuple[str, tuple[int, ...]]:
        """Return (SQL fragment with ? placeholders, parameter tuple) for excluded types."""
        ids = self._get_excluded_ids()
        ph = ",".join("?" * len(ids))
        return f"NOT IN ({ph})", ids

    def _library_filter(self) -> tuple[str, tuple[int, ...]]:
        """Return (SQL fragment, params) for filtering by library.
        Returns empty string/tuple for library_id=1 to preserve existing behavior."""
        if self._library_id == 1:
            return "", ()
        return "AND i.libraryID = ?", (self._library_id,)

    def _connect_from_snapshot(self) -> sqlite3.Connection:
        """Open a throwaway snapshot copy of the database.

        Used when the live database cannot be opened — typically because
        Zotero 10 is running and holds PRAGMA locking_mode=EXCLUSIVE on
        zotero.sqlite. The copy pairs zotero.sqlite with its WAL so SQLite
        replays recent committed writes during recovery; the snapshot is taken
        once per Reader instance and reused until close(). The copy is
        validated with a trivial query and retried once from a fresh copy
        (the live files can change while they are being copied) before giving
        up with an explanatory error.
        """
        last_error: Exception | None = None
        for _attempt in range(2):
            conn: sqlite3.Connection | None = None
            try:
                tmp_db = self._make_snapshot()
                # Read-write so SQLite can recover and replay the copied WAL;
                # the writes recovery makes only touch the throwaway copy.
                conn = sqlite3.connect(str(tmp_db), timeout=5.0)
                conn.row_factory = sqlite3.Row
                conn.execute("SELECT 1 FROM items LIMIT 1")
            except (OSError, sqlite3.Error) as exc:
                if conn is not None:
                    conn.close()
                last_error = exc
                continue
            self._conn = conn
            return conn
        raise sqlite3.OperationalError(
            f"Cannot read the Zotero database: {self._db_path}\n"
            "  Zotero 10 holds an exclusive lock on zotero.sqlite while it is running,\n"
            f"  and reading a snapshot copy failed as well: {last_error}\n"
            "  Close Zotero and try again; if the problem persists, the database may be corrupted."
        ) from last_error

    def _make_snapshot(self) -> Path:
        """Copy zotero.sqlite (and its WAL, if present) into a fresh temp dir.

        The -shm file is deliberately not copied — SQLite rebuilds it during
        WAL recovery. Any previous snapshot for this Reader is discarded.
        Returns the path of the snapshot database.
        """
        if self._tmp_dir_obj is not None:
            self._tmp_dir_obj.cleanup()
        self._tmp_dir_obj = tempfile.TemporaryDirectory(prefix="zot-reader-")
        self._tmp_dir = Path(self._tmp_dir_obj.name)
        tmp_db = self._tmp_dir / self._db_path.name
        shutil.copy2(self._db_path, tmp_db)
        wal = self._db_path.with_suffix(".sqlite-wal")
        if wal.exists():
            shutil.copy2(wal, tmp_db.with_suffix(".sqlite-wal"))
        return tmp_db

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._ft_conn:
            self._ft_conn.close()
            self._ft_conn = None
        self._ft_checked = False
        if hasattr(self, "_tmp_dir_obj") and self._tmp_dir_obj is not None:
            self._tmp_dir_obj.cleanup()
            self._tmp_dir_obj = None
            self._tmp_dir = None

    def __enter__(self) -> ZoteroReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    def get_schema_version(self) -> int | None:
        conn = self._connect()
        row = conn.execute("SELECT version FROM version WHERE schema = 'userdata'").fetchone()
        return row["version"] if row else None

    def check_schema_compatibility(self) -> None:
        version = self.get_schema_version()
        if version and (version < MIN_SCHEMA_VERSION or version > MAX_SCHEMA_VERSION):
            warnings.warn(
                f"Zotero schema version {version} is outside the tested range (100-200). "
                "Some queries may not work correctly.",
                stacklevel=2,
            )

    def resolve_group_library_id(self, group_id: int) -> int | None:
        """Look up the SQLite libraryID for a Zotero group by its public groupID."""
        conn = self._connect()
        row = conn.execute(
            "SELECT libraryID FROM groups WHERE groupID = ?",
            (group_id,),
        ).fetchone()
        return row["libraryID"] if row else None

    def get_item(self, key: str) -> Item | None:
        conn = self._connect()
        lib_sql, lib_params = self._library_filter()
        row = conn.execute(
            "SELECT itemID, itemTypeID, key, dateAdded, dateModified "
            "FROM items i WHERE key = ? AND itemTypeID " + self._get_excluded_sql() + " " + lib_sql,
            (key, *lib_params),
        ).fetchone()
        if row is None:
            return None
        item_id = row["itemID"]
        item_type = conn.execute(
            "SELECT typeName FROM itemTypes WHERE itemTypeID = ?",
            (row["itemTypeID"],),
        ).fetchone()["typeName"]
        fields = self._get_item_fields(conn, item_id)
        creators = self._get_item_creators(conn, item_id)
        tags = self._get_item_tags(conn, item_id)
        collections = self._get_item_collections(conn, item_id)
        return Item(
            key=key,
            item_type=item_type,
            title=fields.get("title", ""),
            creators=creators,
            abstract=fields.get("abstractNote"),
            date=fields.get("date"),
            url=fields.get("url"),
            doi=fields.get("DOI"),
            tags=tags,
            collections=collections,
            date_added=row["dateAdded"],
            date_modified=row["dateModified"],
            extra={k: v for k, v in fields.items() if k not in ("title", "abstractNote", "date", "url", "DOI")},
        )

    def search(
        self,
        query: str,
        collection: str | None = None,
        item_type: str | None = None,
        sort: str | None = None,
        direction: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> SearchResult:
        conn = self._connect()
        item_ids: set[int] = set()
        excl_sql, excl_params = self._excluded_filter()
        lib_sql, lib_params = self._library_filter()

        if query:
            # Tokenize into words and match each word independently across
            # all search dimensions, then intersect (AND).  This lets
            # "Vaswani attention" match an item whose creator is Vaswani
            # and whose title contains "attention" — the words live in
            # different fields.  The original full-phrase LIKE is also kept
            # as an OR contributor so exact phrase matches still rank.
            words = query.split()
            like = f"%{query}%"

            # Per-word match sets for intersection
            word_sets: list[set[int]] = []
            for w in words:
                pat = f"%{w}%"
                w_ids: set[int] = set()

                # Metadata (title / abstract)
                rows = conn.execute(
                    "SELECT DISTINCT i.itemID FROM items i "
                    "JOIN itemData id ON i.itemID = id.itemID "
                    "JOIN itemDataValues iv ON id.valueID = iv.valueID "
                    f"WHERE iv.value LIKE ? AND i.itemTypeID {excl_sql} {lib_sql}",
                    (pat, *excl_params, *lib_params),
                ).fetchall()
                w_ids.update(r["itemID"] for r in rows)

                # Creators
                rows = conn.execute(
                    "SELECT DISTINCT ic.itemID FROM itemCreators ic "
                    "JOIN creators c ON ic.creatorID = c.creatorID "
                    "JOIN items i ON ic.itemID = i.itemID "
                    "WHERE (c.firstName LIKE ? OR c.lastName LIKE ?) "
                    f"AND i.itemTypeID {excl_sql} {lib_sql}",
                    (pat, pat, *excl_params, *lib_params),
                ).fetchall()
                w_ids.update(r["itemID"] for r in rows)

                # Tags
                rows = conn.execute(
                    "SELECT DISTINCT it.itemID FROM itemTags it "
                    "JOIN tags t ON it.tagID = t.tagID "
                    "JOIN items i ON it.itemID = i.itemID "
                    f"WHERE t.name LIKE ? AND i.itemTypeID {excl_sql} {lib_sql}",
                    (pat, *excl_params, *lib_params),
                ).fetchall()
                w_ids.update(r["itemID"] for r in rows)

                # Fulltext (Zotero 10's FTS5 index; skipped when fulltext.sqlite
                # is absent, e.g. pre-Zotero 10 data directories)
                w_ids.update(self._fulltext_parent_ids(w, excl_sql, excl_params, lib_sql, lib_params))

                word_sets.append(w_ids)

            # Intersect: an item must match every word somewhere
            if word_sets:
                item_ids = word_sets[0].copy()
                for ws in word_sets[1:]:
                    item_ids &= ws

            # Also OR in the full-phrase match — exact phrase hits should
            # always appear even if the tokenised intersection missed them.
            rows = conn.execute(
                "SELECT DISTINCT i.itemID FROM items i "
                "JOIN itemData id ON i.itemID = id.itemID "
                "JOIN itemDataValues iv ON id.valueID = iv.valueID "
                f"WHERE iv.value LIKE ? AND i.itemTypeID {excl_sql} {lib_sql}",
                (like, *excl_params, *lib_params),
            ).fetchall()
            item_ids.update(r["itemID"] for r in rows)
        else:
            rows = conn.execute(
                f"SELECT itemID FROM items i WHERE itemTypeID {excl_sql} {lib_sql}",
                (*excl_params, *lib_params),
            ).fetchall()
            item_ids.update(r["itemID"] for r in rows)

        # Validate sort column (prevents SQL injection via f-string interpolation)
        _ALLOWED_SORT_COLS = {"dateAdded", "dateModified", "title", "creator"}
        if sort and sort not in _ALLOWED_SORT_COLS:
            raise ValueError(f"Invalid sort column: {sort}")

        # Filter by collection (accepts key or name)
        if collection:
            col_row = conn.execute(
                "SELECT collectionID FROM collections WHERE libraryID = ? AND (key = ? OR collectionName = ?)",
                (self._library_id, collection, collection),
            ).fetchone()
            if col_row:
                col_items = conn.execute(
                    "SELECT itemID FROM collectionItems WHERE collectionID = ?",
                    (col_row["collectionID"],),
                ).fetchall()
                col_item_ids = {r["itemID"] for r in col_items}
                item_ids &= col_item_ids
            else:
                raise ValueError(
                    f"Collection '{collection}' not found. Use 'zot collection list' to see available names."
                )

        # Filter by item type
        if item_type:
            type_row = conn.execute(
                "SELECT itemTypeID FROM itemTypes WHERE typeName = ?",
                (item_type,),
            ).fetchone()
            if type_row:
                rows = self._batched_in_query(
                    conn,
                    "SELECT itemID FROM items WHERE itemTypeID = ? AND itemID IN ({ph})",
                    sorted(item_ids),
                    (type_row["itemTypeID"],),
                )
                item_ids = {r["itemID"] for r in rows}
            else:
                item_ids = set()

        # Resolve items in batch
        total = len(item_ids)

        if sort and item_ids:
            dir_sql = "DESC" if direction == "desc" else "ASC"
            reverse = direction == "desc"
            id_list = sorted(item_ids)

            if sort in ("dateAdded", "dateModified"):
                if len(id_list) <= _SORT_IN_THRESHOLD:
                    ph = ",".join("?" * len(id_list))
                    rows = conn.execute(
                        f"SELECT itemID FROM items WHERE itemID IN ({ph}) ORDER BY {sort} {dir_sql} LIMIT ? OFFSET ?",
                        (*id_list, limit, offset),
                    ).fetchall()
                    target_ids = [r["itemID"] for r in rows]
                else:
                    # Oversized IN list: fetch sort keys batched, sort in Python
                    pairs = self._batched_in_query(
                        conn,
                        f"SELECT itemID, {sort} AS sort_key FROM items WHERE itemID IN ({{ph}})",
                        id_list,
                    )
                    pairs.sort(key=lambda r: r["sort_key"] or "", reverse=reverse)
                    target_ids = [r["itemID"] for r in pairs[offset : offset + limit]]
            elif sort == "title":
                title_field = conn.execute("SELECT fieldID FROM fields WHERE fieldName = 'title'").fetchone()
                fid = title_field["fieldID"] if title_field else 4
                if len(id_list) <= _SORT_IN_THRESHOLD:
                    ph = ",".join("?" * len(id_list))
                    rows = conn.execute(
                        f"SELECT i.itemID FROM items i "
                        f"LEFT JOIN itemData id_t ON i.itemID = id_t.itemID AND id_t.fieldID = ? "
                        f"LEFT JOIN itemDataValues iv_t ON id_t.valueID = iv_t.valueID "
                        f"WHERE i.itemID IN ({ph}) "
                        f"ORDER BY COALESCE(iv_t.value, '') COLLATE NOCASE {dir_sql} LIMIT ? OFFSET ?",
                        (fid, *id_list, limit, offset),
                    ).fetchall()
                    target_ids = [r["itemID"] for r in rows]
                else:
                    pairs = self._batched_in_query(
                        conn,
                        "SELECT i.itemID, COALESCE(iv_t.value, '') AS sort_key FROM items i "
                        "LEFT JOIN itemData id_t ON i.itemID = id_t.itemID AND id_t.fieldID = ? "
                        "LEFT JOIN itemDataValues iv_t ON id_t.valueID = iv_t.valueID "
                        "WHERE i.itemID IN ({ph})",
                        id_list,
                        (fid,),
                    )
                    pairs.sort(key=lambda r: _nocase_key(r["sort_key"]), reverse=reverse)
                    target_ids = [r["itemID"] for r in pairs[offset : offset + limit]]
            elif sort == "creator":
                if len(id_list) <= _SORT_IN_THRESHOLD:
                    ph = ",".join("?" * len(id_list))
                    rows = conn.execute(
                        f"SELECT i.itemID FROM items i "
                        f"LEFT JOIN itemCreators ic ON i.itemID = ic.itemID AND ic.orderIndex = 0 "
                        f"LEFT JOIN creators c ON ic.creatorID = c.creatorID "
                        f"WHERE i.itemID IN ({ph}) "
                        f"ORDER BY COALESCE(c.lastName, '') COLLATE NOCASE {dir_sql} LIMIT ? OFFSET ?",
                        (*id_list, limit, offset),
                    ).fetchall()
                    target_ids = [r["itemID"] for r in rows]
                else:
                    pairs = self._batched_in_query(
                        conn,
                        "SELECT i.itemID, COALESCE(c.lastName, '') AS sort_key FROM items i "
                        "LEFT JOIN itemCreators ic ON i.itemID = ic.itemID AND ic.orderIndex = 0 "
                        "LEFT JOIN creators c ON ic.creatorID = c.creatorID "
                        "WHERE i.itemID IN ({ph})",
                        id_list,
                    )
                    pairs.sort(key=lambda r: _nocase_key(r["sort_key"]), reverse=reverse)
                    target_ids = [r["itemID"] for r in pairs[offset : offset + limit]]
            else:
                target_ids = sorted(item_ids)[offset : offset + limit]
        else:
            target_ids = sorted(item_ids)[offset : offset + limit]

        items = self._get_items_batch(conn, target_ids) if target_ids else []

        return SearchResult(items=items, total=total, query=query)

    def get_recent_items(
        self,
        since: str,
        sort: str = "dateAdded",
        limit: int = 50,
    ) -> list[Item]:
        """Return items added or modified since the given timestamp."""
        conn = self._connect()
        excl_sql, excl_params = self._excluded_filter()
        lib_sql, lib_params = self._library_filter()
        # Validate sort column (prevents SQL injection via f-string interpolation)
        _ALLOWED_RECENT_SORT = {"dateAdded", "dateModified"}
        if sort and sort not in _ALLOWED_RECENT_SORT:
            raise ValueError(f"Invalid sort column: {sort}")
        col = "dateModified" if sort == "dateModified" else "dateAdded"
        rows = conn.execute(
            f"SELECT itemID FROM items i WHERE {col} >= ? AND itemTypeID {excl_sql} {lib_sql} ORDER BY {col} DESC LIMIT ?",
            (since, *excl_params, *lib_params, limit),
        ).fetchall()
        item_ids = [r["itemID"] for r in rows]
        return self._get_items_batch(conn, item_ids) if item_ids else []

    def get_trash_items(self, limit: int = 50) -> list[Item]:
        """Return items in the trash, ordered by deletion date (newest first)."""
        conn = self._connect()
        excl_sql, excl_params = self._excluded_filter()
        lib_sql, lib_params = self._library_filter()
        rows = conn.execute(
            f"SELECT i.itemID FROM items i "
            f"JOIN deletedItems d ON i.itemID = d.itemID "
            f"WHERE i.itemTypeID {excl_sql} {lib_sql} "
            f"ORDER BY d.dateDeleted DESC LIMIT ?",
            (*excl_params, *lib_params, limit),
        ).fetchall()
        item_ids = [r["itemID"] for r in rows]
        return self._get_items_batch(conn, item_ids) if item_ids else []

    def find_duplicates(
        self,
        strategy: str = "both",
        threshold: float = 0.85,
        limit: int = 50,
    ) -> list[DuplicateGroup]:
        """Find potential duplicate items by DOI and/or title similarity."""
        conn = self._connect()
        excl_sql, excl_params = self._excluded_filter()
        lib_sql, lib_params = self._library_filter()

        # Load items for comparison (cap at 10k most recent)
        rows = conn.execute(
            f"SELECT i.itemID, i.key FROM items i WHERE i.itemTypeID {excl_sql} {lib_sql} ORDER BY i.dateAdded DESC LIMIT 10000",
            (*excl_params, *lib_params),
        ).fetchall()

        item_keys = {r["itemID"]: r["key"] for r in rows}
        item_ids = list(item_keys.keys())
        if not item_ids:
            return []

        groups: list[DuplicateGroup] = []
        seen_group_keys: set[frozenset[str]] = set()

        # --- DOI strategy ---
        if strategy in ("doi", "both"):
            doi_rows = self._batched_in_query(
                conn,
                "SELECT id.itemID, iv.value FROM itemData id "
                "JOIN fields f ON id.fieldID = f.fieldID "
                "JOIN itemDataValues iv ON id.valueID = iv.valueID "
                "WHERE f.fieldName = 'DOI' AND id.itemID IN ({ph}) AND iv.value != ''",
                item_ids,
            )

            doi_map: dict[str, list[int]] = {}
            for r in doi_rows:
                doi_map.setdefault(r["value"].strip().lower(), []).append(r["itemID"])

            for doi_val, ids in doi_map.items():
                if len(ids) < 2:
                    continue
                group_key = frozenset(item_keys[i] for i in ids)
                if group_key in seen_group_keys:
                    continue
                seen_group_keys.add(group_key)
                items = self._get_items_batch(conn, ids)
                if len(items) >= 2:
                    groups.append(DuplicateGroup(items=items, match_type="doi", score=1.0))

        # --- Title strategy ---
        if strategy in ("title", "both"):
            title_rows = self._batched_in_query(
                conn,
                "SELECT id.itemID, iv.value FROM itemData id "
                "JOIN fields f ON id.fieldID = f.fieldID "
                "JOIN itemDataValues iv ON id.valueID = iv.valueID "
                "WHERE f.fieldName = 'title' AND id.itemID IN ({ph})",
                item_ids,
            )

            def _normalize(title: str) -> str:
                t = re.sub(r"[^\w\s]", "", title.lower()).strip()
                return re.sub(r"\s+", " ", t)

            title_items: list[tuple[int, str, str]] = []  # (itemID, original, normalized)
            for r in title_rows:
                title_items.append((r["itemID"], r["value"], _normalize(r["value"])))

            # Group exact normalized matches (O(n))
            norm_groups: dict[str, list[int]] = {}
            for item_id, orig, norm in title_items:
                norm_groups.setdefault(norm, []).append(item_id)

            for norm, ids in norm_groups.items():
                if len(ids) >= 2:
                    group_key = frozenset(item_keys[i] for i in ids)
                    if group_key not in seen_group_keys:
                        seen_group_keys.add(group_key)
                        items = self._get_items_batch(conn, ids)
                        if len(items) >= 2:
                            groups.append(DuplicateGroup(items=items, match_type="title", score=1.0))

            # Fuzzy match singletons only (O(n^2) on singletons)
            singletons = [(item_id, norm) for item_id, _, norm in title_items if len(norm_groups[norm]) == 1]
            matched: set[int] = set()
            for idx, (id_a, norm_a) in enumerate(singletons):
                if id_a in matched:
                    continue
                cluster = [id_a]
                for j in range(idx + 1, len(singletons)):
                    id_b, norm_b = singletons[j]
                    if id_b in matched:
                        continue
                    ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
                    if ratio >= threshold:
                        cluster.append(id_b)
                        matched.add(id_b)
                if len(cluster) >= 2:
                    matched.add(id_a)
                    group_key = frozenset(item_keys[cid] for cid in cluster)
                    if group_key not in seen_group_keys:
                        seen_group_keys.add(group_key)
                        items = self._get_items_batch(conn, cluster)
                        best_score = (
                            max(
                                SequenceMatcher(None, _normalize(items[i].title), _normalize(items[j].title)).ratio()
                                for i in range(len(items))
                                for j in range(i + 1, len(items))
                            )
                            if len(items) >= 2
                            else 0.0
                        )
                        groups.append(DuplicateGroup(items=items, match_type="title", score=round(best_score, 3)))

        return groups[:limit]

    def get_notes(self, key: str) -> list[Note]:
        conn = self._connect()
        parent = conn.execute("SELECT itemID FROM items WHERE key = ?", (key,)).fetchone()
        if parent is None:
            return []
        # A note key reads the note itself; an item key reads its child notes.
        note_row = conn.execute(
            "SELECT i.itemID, i.key, n.note, p.key AS parent_key FROM itemNotes n "
            "JOIN items i ON n.itemID = i.itemID LEFT JOIN items p ON n.parentItemID = p.itemID "
            "WHERE i.key = ?",
            (key,),
        ).fetchone()
        if note_row is not None:
            return self._notes_from_rows(conn, [note_row], parent_key=note_row["parent_key"] or "")
        rows = conn.execute(
            "SELECT i.itemID, i.key, n.note FROM itemNotes n JOIN items i ON n.itemID = i.itemID WHERE n.parentItemID = ?",
            (parent["itemID"],),
        ).fetchall()
        return self._notes_from_rows(conn, rows, parent_key=key)

    def get_standalone_notes(self) -> list[Note]:
        """List all top-level (standalone) notes — note items with no parent."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT i.itemID, i.key, n.note FROM itemNotes n JOIN items i ON n.itemID = i.itemID "
            "WHERE n.parentItemID IS NULL",
        ).fetchall()
        return self._notes_from_rows(conn, rows, parent_key="")

    def _notes_from_rows(self, conn: sqlite3.Connection, rows: list[sqlite3.Row], parent_key: str) -> list[Note]:
        if not rows:
            return []
        # Batch-fetch tags for all note items
        note_ids = [r["itemID"] for r in rows]
        ph = ",".join("?" * len(note_ids))
        tag_rows = conn.execute(
            f"SELECT it.itemID, t.name FROM itemTags it JOIN tags t ON it.tagID = t.tagID WHERE it.itemID IN ({ph})",
            note_ids,
        ).fetchall()
        tags_by_id: dict[int, list[str]] = {}
        for tr in tag_rows:
            tags_by_id.setdefault(tr["itemID"], []).append(tr["name"])
        notes = []
        for r in rows:
            content = self._html_to_markdown(r["note"] or "")
            tags = tags_by_id.get(r["itemID"], [])
            notes.append(Note(key=r["key"], parent_key=parent_key, content=content, tags=tags))
        return notes

    def get_collections(self) -> list[Collection]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT collectionID, collectionName, parentCollectionID, key FROM collections WHERE libraryID = ?",
            (self._library_id,),
        ).fetchall()
        coll_map: dict[int, Collection] = {}
        parent_map: dict[int, int | None] = {}
        for r in rows:
            coll_map[r["collectionID"]] = Collection(
                key=r["key"],
                name=r["collectionName"],
                parent_key=None,
                children=[],
            )
            parent_map[r["collectionID"]] = r["parentCollectionID"]

        for cid, parent_cid in parent_map.items():
            if parent_cid and parent_cid in coll_map:
                coll_map[cid].parent_key = coll_map[parent_cid].key
                coll_map[parent_cid].children.append(coll_map[cid])

        return [c for cid, c in coll_map.items() if parent_map[cid] is None]

    def get_collection_items(self, collection_key: str) -> list[Item]:
        conn = self._connect()
        col_row = conn.execute(
            "SELECT collectionID FROM collections WHERE key = ?",
            (collection_key,),
        ).fetchone()
        if col_row is None:
            return []
        rows = conn.execute(
            "SELECT i.itemID FROM collectionItems ci "
            "JOIN items i ON ci.itemID = i.itemID "
            "WHERE ci.collectionID = ? AND i.itemTypeID " + self._get_excluded_sql(),
            (col_row["collectionID"],),
        ).fetchall()
        if not rows:
            return []
        return self._get_items_batch(conn, [r["itemID"] for r in rows])

    def get_attachments(self, key: str) -> list[Attachment]:
        conn = self._connect()
        parent = conn.execute("SELECT itemID FROM items WHERE key = ?", (key,)).fetchone()
        if parent is None:
            return []
        rows = conn.execute(
            "SELECT i.itemID, i.key, ia.contentType, ia.path "
            "FROM itemAttachments ia "
            "JOIN items i ON ia.itemID = i.itemID "
            "WHERE ia.parentItemID = ? "
            "ORDER BY i.itemID",
            (parent["itemID"],),
        ).fetchall()
        attachments = []
        for r in rows:
            raw_path = r["path"] or ""
            resolved = self._resolver.resolve(r["key"], raw_path)
            if resolved:
                filename = resolved.name
            elif raw_path.startswith("storage:"):
                parts = raw_path.replace("storage:", "").split("/")
                filename = parts[-1] if parts else ""
            else:
                filename = raw_path.split("/")[-1] if raw_path else ""
            attachments.append(
                Attachment(
                    key=r["key"],
                    parent_key=key,
                    filename=filename,
                    content_type=r["contentType"] or "",
                    path=resolved,
                    tags=self._get_item_tags(conn, r["itemID"]),
                )
            )
        return attachments

    def get_pdf_attachments(self, key: str, skip_tags: set[str] | None = None) -> list[Attachment]:
        """Return every PDF attachment for a parent item, in attachment order.

        Modern items frequently carry more than one PDF (e.g. the article plus a
        supplementary appendix or a translated copy). This returns all of them so
        callers can hand the full set to an agent; use `get_pdf_attachment` when
        only the first is needed. `skip_tags` excludes attachments carrying a
        marker tag like `skip-index`.
        """
        result: list[Attachment] = []
        for att in self.get_attachments(key):
            if att.content_type != "application/pdf":
                continue
            if skip_tags and skip_tags.intersection(att.tags):
                continue
            result.append(att)
        return result

    def get_pdf_attachment(self, key: str, skip_tags: set[str] | None = None) -> Attachment | None:
        """Return the first PDF attachment, skipping any tagged with a tag in `skip_tags`.

        `skip_tags` lets callers exclude redundant attachments such as
        machine-translated copies or slides that carry a marker tag like
        `skip-index`.
        """
        attachments = self.get_pdf_attachments(key, skip_tags=skip_tags)
        return attachments[0] if attachments else None

    def _fulltext_parent_ids(
        self,
        term: str,
        excl_sql: str,
        excl_params: tuple[int, ...],
        lib_sql: str,
        lib_params: tuple[int, ...],
    ) -> set[int]:
        """Parent itemIDs whose attachments' indexed full text matches `term`.

        The term is resolved to an FTS5 MATCH clause the way Zotero 10 builds
        it (see core.fts); terms the index can't answer and data directories
        without fulltext.sqlite yield an empty set.
        """
        ft = self._connect_fulltext()
        if ft is None:
            return set()
        clause = content_match_clause(term)
        if clause is None:
            return set()
        table, match = clause
        rows = ft.execute(f"SELECT rowid FROM {table} WHERE {table} MATCH ?", (match,)).fetchall()
        if not rows:
            return set()
        att_ids = [r["rowid"] for r in rows]
        conn = self._connect()
        parent_ids: set[int] = set()
        for batch in _in_batches(att_ids):
            placeholders = ",".join("?" * len(batch))
            prows = conn.execute(
                "SELECT DISTINCT ia.parentItemID FROM itemAttachments ia "
                "JOIN items i ON ia.parentItemID = i.itemID "
                f"WHERE ia.itemID IN ({placeholders}) AND ia.parentItemID IS NOT NULL "
                f"AND i.itemTypeID {excl_sql} {lib_sql}",
                (*batch, *excl_params, *lib_params),
            ).fetchall()
            parent_ids.update(r["parentItemID"] for r in prows)
        return parent_ids

    def fulltext_candidates(self, collection_key: str | None = None) -> list[tuple[str, int]]:
        """Return (parent_item_key, attachment_itemID) for indexed attachments.

        "Indexed" follows Zotero 10's semantics: the attachment has a
        fulltextIndexState row at the current content index version (see
        core.fts.CONTENT_INDEX_VERSION). Used by `core.rank` for full-text
        ranking over fulltext.sqlite. `collection_key` restricts the scope to
        that collection's items. Returns [] when there is no fulltext.sqlite
        (pre-Zotero 10 data directories) or nothing in scope is indexed.
        """
        ft = self._connect_fulltext()
        if ft is None:
            return []
        indexed = {
            r["itemID"]
            for r in ft.execute(
                "SELECT itemID FROM fulltextIndexState WHERE version >= ?",
                (CONTENT_INDEX_VERSION,),
            ).fetchall()
        }
        if not indexed:
            return []
        conn = self._connect()
        col_sql = ""
        col_params: tuple = ()
        if collection_key is not None:
            col_row = conn.execute(
                "SELECT collectionID FROM collections WHERE libraryID = ? AND key = ?",
                (self._library_id, collection_key),
            ).fetchone()
            if col_row is None:
                return []
            col_sql = " AND i.itemID IN (SELECT itemID FROM collectionItems WHERE collectionID = ?)"
            col_params = (col_row["collectionID"],)
        rows = conn.execute(
            "SELECT i.key AS parent_key, ia.itemID AS att_id "
            "FROM itemAttachments ia "
            "JOIN items i ON ia.parentItemID = i.itemID "
            f"WHERE ia.parentItemID IS NOT NULL AND i.itemTypeID {self._get_excluded_sql()} "
            f"AND i.libraryID = ?{col_sql}",
            (self._library_id, *col_params),
        ).fetchall()
        return [(r["parent_key"], r["att_id"]) for r in rows if r["att_id"] in indexed]

    def find_orphan_attachments(self) -> list[OrphanAttachment]:
        """Find storage-backed attachments whose file is missing from local storage.

        These are the records that make Zotero show "the attached file could not
        be found" — typically created by a Web-API upload that landed the file
        in cloud storage only. Each is classified (see `OrphanAttachment`) so a
        caller can safely delete the truly dead ones while leaving merely
        not-yet-downloaded files alone.
        """
        conn = self._connect()
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(itemAttachments)")}
        has_sync = "syncState" in cols
        has_hash = "storageHash" in cols
        extra = (", ia.syncState" if has_sync else "") + (", ia.storageHash" if has_hash else "")
        rows = conn.execute(
            "SELECT i.key, ia.parentItemID, ia.contentType, ia.path" + extra + " "
            "FROM itemAttachments ia JOIN items i ON ia.itemID = i.itemID "
            "WHERE ia.path LIKE 'storage:%' "
            "ORDER BY i.itemID"
        ).fetchall()

        orphans: list[OrphanAttachment] = []
        for r in rows:
            raw_path = r["path"] or ""
            resolved = self._resolver.resolve(r["key"], raw_path)
            if resolved and resolved.exists():
                continue  # file is present locally — not an orphan

            sync_state = r["syncState"] if has_sync else None
            storage_hash = r["storageHash"] if has_hash else None
            if not has_sync and not has_hash:
                classification = "unknown"
            elif storage_hash or sync_state in (_SYNC_STATE_TO_DOWNLOAD, _SYNC_STATE_FORCE_DOWNLOAD):
                classification = "recoverable"
            else:
                classification = "dead"

            parent_key = None
            parent_title = None
            if r["parentItemID"] is not None:
                prow = conn.execute("SELECT key FROM items WHERE itemID = ?", (r["parentItemID"],)).fetchone()
                if prow:
                    parent_key = prow["key"]
                    parent_item = self.get_item(parent_key)
                    parent_title = parent_item.title if parent_item else None

            orphans.append(
                OrphanAttachment(
                    attachment_key=r["key"],
                    filename=raw_path.replace("storage:", "").split("/")[-1] if raw_path else "",
                    content_type=r["contentType"] or "",
                    classification=classification,
                    expected_path=str(resolved) if resolved else None,
                    parent_key=parent_key,
                    parent_title=parent_title,
                )
            )
        return orphans

    def get_arxiv_preprints(
        self,
        collection: str | None = None,
        limit: int = 200,
    ) -> list[Item]:
        """Find preprint items (arXiv, bioRxiv, medRxiv) by URL, DOI, extra, or itemType."""
        conn = self._connect()
        excl_sql, excl_params = self._excluded_filter()
        lib_sql, lib_params = self._library_filter()

        # Find items with arXiv or bioRxiv/medRxiv references in URL, DOI, or extra fields
        rows = conn.execute(
            "SELECT DISTINCT id.itemID FROM itemData id "
            "JOIN fields f ON id.fieldID = f.fieldID "
            "JOIN itemDataValues iv ON id.valueID = iv.valueID "
            "JOIN items i ON id.itemID = i.itemID "
            f"WHERE f.fieldName IN ('url', 'DOI', 'extra') "
            f"AND (iv.value LIKE '%arxiv%' OR iv.value LIKE '%biorxiv%' OR iv.value LIKE '%medrxiv%' "
            f"OR iv.value LIKE '%10.1101/%') "
            f"AND i.itemTypeID {excl_sql} {lib_sql}",
            (*excl_params, *lib_params),
        ).fetchall()
        item_ids = {r["itemID"] for r in rows}

        # Also find all items with itemType = preprint
        preprint_type = conn.execute("SELECT itemTypeID FROM itemTypes WHERE typeName = 'preprint'").fetchone()
        if preprint_type:
            rows = conn.execute(
                f"SELECT i.itemID FROM items i WHERE i.itemTypeID = ? {lib_sql}",
                (preprint_type["itemTypeID"], *lib_params),
            ).fetchall()
            for r in rows:
                item_ids.add(r["itemID"])

        if not item_ids:
            return []

        # Filter by collection if specified
        if collection:
            col_row = conn.execute(
                "SELECT collectionID FROM collections WHERE libraryID = ? AND (key = ? OR collectionName = ?)",
                (self._library_id, collection, collection),
            ).fetchone()
            if col_row:
                col_items = conn.execute(
                    "SELECT itemID FROM collectionItems WHERE collectionID = ?",
                    (col_row["collectionID"],),
                ).fetchall()
                item_ids &= {r["itemID"] for r in col_items}
            else:
                raise ValueError(
                    f"Collection '{collection}' not found. Use 'zot collection list' to see available names."
                )

        target_ids = sorted(item_ids)[:limit]
        return self._get_items_batch(conn, target_ids) if target_ids else []

    def get_stats(self) -> dict:
        """Return library statistics."""
        conn = self._connect()
        lib_sql, lib_params = self._library_filter()
        # Total items (excluding attachments and notes)
        total = conn.execute(
            "SELECT COUNT(*) as cnt FROM items i WHERE itemTypeID " + self._get_excluded_sql() + " " + lib_sql,
            lib_params,
        ).fetchone()["cnt"]

        # Items by type
        type_rows = conn.execute(
            "SELECT t.typeName, COUNT(*) as cnt FROM items i "
            "JOIN itemTypes t ON i.itemTypeID = t.itemTypeID "
            "WHERE i.itemTypeID " + self._get_excluded_sql() + " " + lib_sql + " "
            "GROUP BY t.typeName ORDER BY cnt DESC",
            lib_params,
        ).fetchall()
        by_type = {r["typeName"]: r["cnt"] for r in type_rows}

        # Top tags
        tag_rows = conn.execute(
            "SELECT t.name, COUNT(*) as cnt FROM itemTags it "
            "JOIN tags t ON it.tagID = t.tagID "
            "JOIN items i ON it.itemID = i.itemID "
            "WHERE 1=1 " + lib_sql + " "
            "GROUP BY t.name ORDER BY cnt DESC LIMIT 20",
            lib_params,
        ).fetchall()
        top_tags = {r["name"]: r["cnt"] for r in tag_rows}

        # Collections with item counts
        coll_rows = conn.execute(
            "SELECT c.collectionName, COUNT(ci.itemID) as cnt "
            "FROM collections c "
            "LEFT JOIN collectionItems ci ON c.collectionID = ci.collectionID "
            "WHERE 1=1 " + lib_sql.replace("i.libraryID", "c.libraryID") + " "
            "GROUP BY c.collectionName ORDER BY cnt DESC",
            lib_params,
        ).fetchall()
        collections = {r["collectionName"]: r["cnt"] for r in coll_rows}

        # Attachments
        pdf_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM itemAttachments a "
            "JOIN items i ON a.itemID = i.itemID "
            "WHERE a.contentType = 'application/pdf' " + lib_sql,
            lib_params,
        ).fetchone()["cnt"]

        # Notes count
        notes_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM itemNotes n JOIN items i ON n.itemID = i.itemID WHERE 1=1 " + lib_sql,
            lib_params,
        ).fetchone()["cnt"]

        return {
            "total_items": total,
            "by_type": by_type,
            "top_tags": top_tags,
            "collections": collections,
            "pdf_attachments": pdf_count,
            "notes": notes_count,
        }

    def export_citation(self, key: str, fmt: str = "bibtex") -> str | None:
        item = self.get_item(key)
        if item is None:
            return None
        if fmt == "bibtex":
            return self._to_bibtex(item)
        if fmt in ("csl", "csl-json", "json"):
            return self._to_csl_json(item)
        if fmt == "ris":
            return self._to_ris(item)
        return None

    def get_related_items(self, key: str, limit: int = 20) -> list[Item]:
        conn = self._connect()
        parent = conn.execute("SELECT itemID FROM items WHERE key = ?", (key,)).fetchone()
        if parent is None:
            return []
        item_id = parent["itemID"]
        related_ids: dict[int, int] = {}

        # Explicit relations
        rows = conn.execute(
            "SELECT object FROM itemRelations WHERE itemID = ? AND predicateID = 1",
            (item_id,),
        ).fetchall()
        for r in rows:
            obj = r["object"]
            rel_key = obj.rsplit("/", 1)[-1] if "/" in obj else obj
            rel_row = conn.execute("SELECT itemID FROM items WHERE key = ?", (rel_key,)).fetchone()
            if rel_row:
                related_ids[rel_row["itemID"]] = related_ids.get(rel_row["itemID"], 0) + 100

        # Implicit: shared collections
        my_cols = {
            r["collectionID"]
            for r in conn.execute("SELECT collectionID FROM collectionItems WHERE itemID = ?", (item_id,)).fetchall()
        }
        if my_cols:
            placeholders = ",".join("?" * len(my_cols))
            rows = conn.execute(
                f"SELECT itemID, COUNT(*) as cnt FROM collectionItems "
                f"WHERE collectionID IN ({placeholders}) AND itemID != ? "
                f"GROUP BY itemID",
                (*my_cols, item_id),
            ).fetchall()
            for r in rows:
                related_ids[r["itemID"]] = related_ids.get(r["itemID"], 0) + r["cnt"]

        # Implicit: shared tags (2+ overlap)
        my_tags = {
            r["tagID"] for r in conn.execute("SELECT tagID FROM itemTags WHERE itemID = ?", (item_id,)).fetchall()
        }
        if my_tags:
            placeholders = ",".join("?" * len(my_tags))
            rows = conn.execute(
                f"SELECT itemID, COUNT(*) as cnt FROM itemTags "
                f"WHERE tagID IN ({placeholders}) AND itemID != ? "
                f"GROUP BY itemID HAVING cnt >= 2",
                (*my_tags, item_id),
            ).fetchall()
            for r in rows:
                related_ids[r["itemID"]] = related_ids.get(r["itemID"], 0) + r["cnt"] * 5

        sorted_ids = sorted(related_ids, key=lambda x: related_ids[x], reverse=True)[:limit]
        items = []
        for rid in sorted_ids:
            key_row = conn.execute("SELECT key FROM items WHERE itemID = ?", (rid,)).fetchone()
            if key_row:
                item = self.get_item(key_row["key"])
                if item:
                    items.append(item)
        return items

    # --- Private helpers ---

    @staticmethod
    def _batched_in_query(
        conn: sqlite3.Connection,
        sql: str,
        ids: list[int],
        prefix_params: tuple[Any, ...] = (),
    ) -> list[sqlite3.Row]:
        """Run a query containing one ``IN ({ph})`` placeholder, batched over ``ids``.

        SQLITE_MAX_VARIABLE_NUMBER can be as low as 999 on older SQLite builds,
        so large id lists are split into batches. ``prefix_params`` bind before
        each id batch.
        """
        out: list[sqlite3.Row] = []
        for batch in _in_batches(ids):
            ph = ",".join("?" * len(batch))
            out.extend(conn.execute(sql.format(ph=ph), (*prefix_params, *batch)).fetchall())
        return out

    def _get_items_batch(self, conn: sqlite3.Connection, item_ids: list[int]) -> list[Item]:
        """Resolve multiple item IDs to Items using bulk queries instead of N+1."""
        if not item_ids:
            return []

        # Fetch base item rows
        rows = self._batched_in_query(
            conn,
            f"SELECT itemID, itemTypeID, key, dateAdded, dateModified "
            f"FROM items WHERE itemID IN ({{ph}}) AND itemTypeID {self._get_excluded_sql()}",
            item_ids,
        )
        if not rows:
            return []

        id_to_row = {r["itemID"]: r for r in rows}
        valid_ids = list(id_to_row.keys())

        # Batch fetch item types
        type_ids = list({r["itemTypeID"] for r in rows})
        type_ph = ",".join("?" * len(type_ids))
        type_rows = conn.execute(
            f"SELECT itemTypeID, typeName FROM itemTypes WHERE itemTypeID IN ({type_ph})",
            type_ids,
        ).fetchall()
        type_map = {r["itemTypeID"]: r["typeName"] for r in type_rows}

        # Batch fetch fields
        field_rows = self._batched_in_query(
            conn,
            "SELECT id.itemID, f.fieldName, iv.value FROM itemData id "
            "JOIN fields f ON id.fieldID = f.fieldID "
            "JOIN itemDataValues iv ON id.valueID = iv.valueID "
            "WHERE id.itemID IN ({ph})",
            valid_ids,
        )
        fields_map: dict[int, dict[str, str]] = {}
        for r in field_rows:
            fields_map.setdefault(r["itemID"], {})[r["fieldName"]] = r["value"]

        # Batch fetch creators
        creator_rows = self._batched_in_query(
            conn,
            "SELECT ic.itemID, c.firstName, c.lastName, ct.creatorType "
            "FROM itemCreators ic "
            "JOIN creators c ON ic.creatorID = c.creatorID "
            "JOIN creatorTypes ct ON ic.creatorTypeID = ct.creatorTypeID "
            "WHERE ic.itemID IN ({ph}) ORDER BY ic.itemID, ic.orderIndex",
            valid_ids,
        )
        creators_map: dict[int, list[Creator]] = {}
        for r in creator_rows:
            creators_map.setdefault(r["itemID"], []).append(
                Creator(r["firstName"] or "", r["lastName"] or "", r["creatorType"])
            )

        # Batch fetch tags
        tag_rows = self._batched_in_query(
            conn,
            "SELECT it.itemID, t.name FROM itemTags it JOIN tags t ON it.tagID = t.tagID WHERE it.itemID IN ({ph})",
            valid_ids,
        )
        tags_map: dict[int, list[str]] = {}
        for r in tag_rows:
            tags_map.setdefault(r["itemID"], []).append(r["name"])

        # Batch fetch collections
        coll_rows = self._batched_in_query(
            conn,
            "SELECT ci.itemID, c.key FROM collectionItems ci "
            "JOIN collections c ON ci.collectionID = c.collectionID "
            "WHERE ci.itemID IN ({ph})",
            valid_ids,
        )
        colls_map: dict[int, list[str]] = {}
        for r in coll_rows:
            colls_map.setdefault(r["itemID"], []).append(r["key"])

        # Assemble items in original order
        items: list[Item] = []
        for item_id in item_ids:
            if item_id not in id_to_row:
                continue
            row = id_to_row[item_id]
            fields = fields_map.get(item_id, {})
            items.append(
                Item(
                    key=row["key"],
                    item_type=type_map.get(row["itemTypeID"], "unknown"),
                    title=fields.get("title", ""),
                    creators=creators_map.get(item_id, []),
                    abstract=fields.get("abstractNote"),
                    date=fields.get("date"),
                    url=fields.get("url"),
                    doi=fields.get("DOI"),
                    tags=tags_map.get(item_id, []),
                    collections=colls_map.get(item_id, []),
                    date_added=row["dateAdded"],
                    date_modified=row["dateModified"],
                    extra={k: v for k, v in fields.items() if k not in ("title", "abstractNote", "date", "url", "DOI")},
                )
            )
        return items

    def _get_item_fields(self, conn: sqlite3.Connection, item_id: int) -> dict[str, str]:
        rows = conn.execute(
            "SELECT f.fieldName, iv.value FROM itemData id "
            "JOIN fields f ON id.fieldID = f.fieldID "
            "JOIN itemDataValues iv ON id.valueID = iv.valueID "
            "WHERE id.itemID = ?",
            (item_id,),
        ).fetchall()
        return {r["fieldName"]: r["value"] for r in rows}

    def _get_item_creators(self, conn: sqlite3.Connection, item_id: int) -> list[Creator]:
        rows = conn.execute(
            "SELECT c.firstName, c.lastName, ct.creatorType "
            "FROM itemCreators ic "
            "JOIN creators c ON ic.creatorID = c.creatorID "
            "JOIN creatorTypes ct ON ic.creatorTypeID = ct.creatorTypeID "
            "WHERE ic.itemID = ? ORDER BY ic.orderIndex",
            (item_id,),
        ).fetchall()
        return [Creator(r["firstName"] or "", r["lastName"] or "", r["creatorType"]) for r in rows]

    def _get_item_tags(self, conn: sqlite3.Connection, item_id: int) -> list[str]:
        rows = conn.execute(
            "SELECT t.name FROM itemTags it JOIN tags t ON it.tagID = t.tagID WHERE it.itemID = ?",
            (item_id,),
        ).fetchall()
        return [r["name"] for r in rows]

    def _get_item_collections(self, conn: sqlite3.Connection, item_id: int) -> list[str]:
        rows = conn.execute(
            "SELECT c.key FROM collectionItems ci "
            "JOIN collections c ON ci.collectionID = c.collectionID "
            "WHERE ci.itemID = ?",
            (item_id,),
        ).fetchall()
        return [r["key"] for r in rows]

    @staticmethod
    def _escape_bibtex(value: str) -> str:
        """Escape special LaTeX/BibTeX characters in a field value."""
        for char, escaped in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")):
            value = value.replace(char, escaped)
        return value

    @staticmethod
    def _to_bibtex(item: Item) -> str:
        type_map = {"journalArticle": "article", "book": "book", "thesis": "phdthesis"}
        bib_type = type_map.get(item.item_type, "misc")
        cite_key = item.key.lower()
        esc = ZoteroReader._escape_bibtex
        authors = " and ".join(
            f"{esc(c.last_name)}, {esc(c.first_name)}" for c in item.creators if c.creator_type == "author"
        )
        lines = [f"@{bib_type}{{{cite_key},"]
        if item.title:
            lines.append(f"  title = {{{esc(item.title)}}},")
        if authors:
            lines.append(f"  author = {{{authors}}},")
        if item.date:
            lines.append(f"  year = {{{item.date}}},")
        if item.doi:
            lines.append(f"  doi = {{{item.doi}}},")
        if item.url:
            lines.append(f"  url = {{{item.url}}},")
        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def _to_csl_json(item: Item) -> str:
        """Convert an Item to CSL-JSON format (single item, not array)."""
        type_map = {
            "journalArticle": "article-journal",
            "book": "book",
            "bookSection": "chapter",
            "conferencePaper": "paper-conference",
            "thesis": "thesis",
            "report": "report",
            "webpage": "webpage",
            "preprint": "article",
        }
        csl: dict = {
            "id": item.key,
            "type": type_map.get(item.item_type, "article"),
            "title": item.title,
        }
        if item.creators:
            csl["author"] = [
                {"family": c.last_name, "given": c.first_name} for c in item.creators if c.creator_type == "author"
            ]
        if item.date:
            csl["issued"] = {"raw": item.date}
        if item.abstract:
            csl["abstract"] = item.abstract
        if item.doi:
            csl["DOI"] = item.doi
        if item.url:
            csl["URL"] = item.url
        return json.dumps(csl, indent=2, ensure_ascii=False)

    @staticmethod
    def _to_ris(item: Item) -> str:
        """Convert an Item to RIS format."""
        type_map = {
            "journalArticle": "JOUR",
            "book": "BOOK",
            "bookSection": "CHAP",
            "conferencePaper": "CONF",
            "thesis": "THES",
            "report": "RPRT",
            "webpage": "ELEC",
            "preprint": "JOUR",
            "patent": "PAT",
            "newspaperArticle": "NEWS",
            "magazineArticle": "MGZN",
        }
        lines = [f"TY  - {type_map.get(item.item_type, 'GEN')}"]
        if item.title:
            lines.append(f"TI  - {item.title}")
        for c in item.creators:
            if c.creator_type == "author":
                lines.append(f"AU  - {c.last_name}, {c.first_name}")
        if item.date:
            lines.append(f"PY  - {item.date}")
        if item.abstract:
            lines.append(f"AB  - {item.abstract}")
        if item.doi:
            lines.append(f"DO  - {item.doi}")
        if item.url:
            lines.append(f"UR  - {item.url}")
        for tag in item.tags:
            lines.append(f"KW  - {tag}")
        lines.append("ER  - ")
        return "\n".join(lines)

    @staticmethod
    def _html_to_markdown(html: str) -> str:
        from markdownify import markdownify as md

        return md(html, strip=["img"]).strip()
