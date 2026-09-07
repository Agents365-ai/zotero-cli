"""Tests for stats and open commands."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from zotero_cli_cc.cli import main


class TestStatsCmd:
    def test_stats_human(self, test_db_path: Path):
        runner = CliRunner()
        result = runner.invoke(main, ["stats"], env={"ZOT_DATA_DIR": str(test_db_path.parent), "ZOT_FORMAT": "table"})
        assert result.exit_code == 0
        assert "Total items:" in result.output
        assert "Items by type:" in result.output
        assert "Collections" in result.output
        assert "Top tags" in result.output

    def test_stats_json(self, test_db_path: Path):
        runner = CliRunner()
        result = runner.invoke(
            main, ["--json", "stats"], env={"ZOT_DATA_DIR": str(test_db_path.parent), "ZOT_FORMAT": "table"}
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["ok"] is True
        stats = data["data"]
        assert "total_items" in stats
        assert "by_type" in stats
        assert "top_tags" in stats
        assert "collections" in stats
        assert "pdf_attachments" in stats
        assert "notes" in stats
        assert stats["total_items"] > 0

    def test_stats_group_library(self, test_db_path: Path):
        result = CliRunner().invoke(
            main,
            ["--json", "--library", "group:99999", "stats"],
            env={"ZOT_DATA_DIR": str(test_db_path.parent)},
        )
        assert result.exit_code == 0
        stats = json.loads(result.output)["data"]
        assert stats["pdf_attachments"] == 1
        assert stats["notes"] == 0
        assert stats["top_tags"] == {}
        assert stats["collections"] == {"Group Papers": 1}


class TestOpenCmd:
    def test_open_nonexistent_item(self, test_db_path: Path):
        runner = CliRunner()
        result = runner.invoke(
            main, ["open", "NONEXIST"], env={"ZOT_DATA_DIR": str(test_db_path.parent), "ZOT_FORMAT": "table"}
        )
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_open_no_pdf(self, test_db_path: Path):
        runner = CliRunner()
        result = runner.invoke(
            main, ["open", "DEEP003"], env={"ZOT_DATA_DIR": str(test_db_path.parent), "ZOT_FORMAT": "table"}
        )
        assert result.exit_code != 0
        assert "No PDF" in result.output

    @patch("zotero_cli_cc.commands.open_cmd._open_path")
    def test_open_url(self, mock_open, test_db_path: Path):
        runner = CliRunner()
        result = runner.invoke(
            main, ["open", "--url", "ATTN001"], env={"ZOT_DATA_DIR": str(test_db_path.parent), "ZOT_FORMAT": "table"}
        )
        assert result.exit_code == 0
        assert "Opening" in result.output
        mock_open.assert_called_once()

    def test_open_url_no_url(self, test_db_path: Path):
        runner = CliRunner()
        # DEEP003 is a book, might not have URL - test the error path
        result = runner.invoke(
            main, ["open", "--url", "DEEP003"], env={"ZOT_DATA_DIR": str(test_db_path.parent), "ZOT_FORMAT": "table"}
        )
        assert result.exit_code != 0


class TestGetStats:
    @pytest.mark.parametrize(
        ("library_id", "pdfs", "notes", "tags", "collections"),
        [
            (
                1,
                6,
                2,
                {"transformer": 3, "attention": 1, "NLP": 1, "scaling": 1, "skip-index": 1, "protein": 1},
                {"Machine Learning": 5, "Transformers": 1, "Empty": 0},
            ),
            (2, 2, 1, {"transformer": 1, "protein": 1}, {"Machine Learning": 1, "Empty": 0}),
            (3, 0, 0, {}, {}),
        ],
    )
    def test_library_scope(self, test_db_path: Path, tmp_path: Path, library_id, pdfs, notes, tags, collections):
        from zotero_cli_cc.core.reader import ZoteroReader

        db_path = tmp_path / "zotero.sqlite"
        shutil.copyfile(test_db_path, db_path)
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript("""
                INSERT INTO libraries VALUES (3, 'group', 1, 1);
                INSERT INTO items (itemID, itemTypeID, libraryID, key) VALUES
                    (16, 26, 2, 'GRPNOTE1'), (17, 14, 2, 'GRPPDF01'), (18, 14, 2, 'GRPHTML1');
                INSERT INTO itemNotes VALUES (16, NULL, '<p>Standalone note</p>', 'Group note');
                INSERT INTO itemAttachments VALUES
                    (17, NULL, 0, 'application/pdf', NULL, 'storage:standalone.pdf'),
                    (18, 9, 0, 'text/html', NULL, 'storage:snapshot.html');
                INSERT INTO tags VALUES (6, 'protein');
                INSERT INTO itemTags VALUES (9, 1, 0), (16, 6, 0);
                UPDATE collections SET collectionName = 'Machine Learning' WHERE collectionID = 3;
                INSERT INTO collections VALUES (4, 'Empty', NULL, 2, 'GRPEMPTY');
            """)
        finally:
            conn.close()

        with ZoteroReader(db_path, library_id=library_id) as reader:
            stats = reader.get_stats()
        assert stats["pdf_attachments"] == pdfs
        assert stats["notes"] == notes
        assert stats["top_tags"] == tags
        assert stats["collections"] == collections

    def test_get_stats(self, test_db_path: Path):
        from zotero_cli_cc.core.reader import ZoteroReader

        reader = ZoteroReader(test_db_path)
        try:
            stats = reader.get_stats()
            assert stats["total_items"] > 0
            assert isinstance(stats["by_type"], dict)
            assert isinstance(stats["top_tags"], dict)
            assert isinstance(stats["collections"], dict)
            assert stats["pdf_attachments"] >= 0
            assert stats["notes"] >= 0
        finally:
            reader.close()
