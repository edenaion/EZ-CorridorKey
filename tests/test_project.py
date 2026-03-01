"""Tests for backend.project module — project folder creation and metadata."""
import json
import os
import tempfile
from unittest.mock import patch

import pytest

from backend.project import (
    sanitize_stem,
    create_project,
    write_project_json,
    read_project_json,
    get_display_name,
    set_display_name,
    is_video_file,
)


class TestSanitizeStem:
    def test_basic(self):
        assert sanitize_stem("Woman_Jumps_For_Joy.mp4") == "Woman_Jumps_For_Joy"

    def test_spaces(self):
        assert sanitize_stem("my cool video.mp4") == "my_cool_video"

    def test_special_chars(self):
        assert sanitize_stem("file (1) [copy].mov") == "file_1_copy"

    def test_collapses_underscores(self):
        assert sanitize_stem("a___b___c.mp4") == "a_b_c"

    def test_truncates_long(self):
        long_name = "a" * 100 + ".mp4"
        result = sanitize_stem(long_name, max_len=60)
        assert len(result) == 60

    def test_strips_leading_trailing(self):
        assert sanitize_stem("___name___.mp4") == "name"


class TestIsVideoFile:
    def test_mp4(self):
        assert is_video_file("test.mp4") is True

    def test_mov(self):
        assert is_video_file("test.MOV") is True

    def test_png(self):
        assert is_video_file("test.png") is False

    def test_no_extension(self):
        assert is_video_file("testfile") is False


class TestProjectJson:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"version": 1, "display_name": "Test Project"}
            write_project_json(tmpdir, data)

            result = read_project_json(tmpdir)
            assert result == data

    def test_read_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert read_project_json(tmpdir) is None

    def test_read_corrupt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "project.json")
            with open(path, "w") as f:
                f.write("not json")
            assert read_project_json(tmpdir) is None

    def test_atomic_write(self):
        """Write should not leave .tmp files on success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            write_project_json(tmpdir, {"test": True})
            files = os.listdir(tmpdir)
            assert "project.json" in files
            assert "project.json.tmp" not in files


class TestDisplayName:
    def test_get_from_project_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_project_json(tmpdir, {"display_name": "My Project"})
            assert get_display_name(tmpdir) == "My Project"

    def test_fallback_to_folder_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            name = get_display_name(tmpdir)
            assert name == os.path.basename(tmpdir)

    def test_set_display_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_project_json(tmpdir, {"version": 1, "display_name": "Old"})
            set_display_name(tmpdir, "New Name")
            assert get_display_name(tmpdir) == "New Name"

    def test_set_creates_json_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_display_name(tmpdir, "New Name")
            assert get_display_name(tmpdir) == "New Name"


class TestCreateProject:
    def test_creates_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy video
            video = os.path.join(tmpdir, "Woman_Jumps.mp4")
            with open(video, "wb") as f:
                f.write(b"\x00" * 100)

            with patch("backend.project.projects_root", return_value=tmpdir):
                project_dir = create_project(video)

            assert os.path.isdir(project_dir)
            assert os.path.isdir(os.path.join(project_dir, "Source"))

            # Video copied
            source_files = os.listdir(os.path.join(project_dir, "Source"))
            assert "Woman_Jumps.mp4" in source_files

            # project.json written
            data = read_project_json(project_dir)
            assert data is not None
            assert data["version"] == 1
            assert "Woman" in data["display_name"]

    def test_folder_naming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video = os.path.join(tmpdir, "test.mp4")
            with open(video, "wb") as f:
                f.write(b"\x00" * 100)

            with patch("backend.project.projects_root", return_value=tmpdir):
                project_dir = create_project(video)

            folder_name = os.path.basename(project_dir)
            # Should start with YYMMDD_HHMMSS_
            parts = folder_name.split("_")
            assert len(parts) >= 3
            # Date part: YYMMDD (6 digits)
            assert len(parts[0]) == 6
            assert parts[0].isdigit()

    def test_rapid_import_deduplicates(self):
        """Rapid duplicate import creates separate project folders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video = os.path.join(tmpdir, "test.mp4")
            with open(video, "wb") as f:
                f.write(b"\x00" * 100)

            with patch("backend.project.projects_root", return_value=tmpdir):
                project_dir1 = create_project(video)
                project_dir2 = create_project(video)

            # Should get different folders (dedup suffix added)
            assert project_dir1 != project_dir2
            # Both should have the source video
            assert os.path.isfile(os.path.join(project_dir1, "Source", "test.mp4"))
            assert os.path.isfile(os.path.join(project_dir2, "Source", "test.mp4"))
