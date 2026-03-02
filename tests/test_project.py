"""Tests for backend.project module — project folder creation and metadata."""
import json
import os
import tempfile
from unittest.mock import patch

import pytest

from backend.project import (
    sanitize_stem,
    create_project,
    add_clips_to_project,
    get_clip_dirs,
    is_v2_project,
    write_project_json,
    read_project_json,
    write_clip_json,
    read_clip_json,
    get_display_name,
    set_display_name,
    save_in_out_range,
    load_in_out_range,
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


class TestClipJson:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"source": {"filename": "test.mp4"}}
            write_clip_json(tmpdir, data)
            result = read_clip_json(tmpdir)
            assert result == data

    def test_read_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert read_clip_json(tmpdir) is None

    def test_read_corrupt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "clip.json")
            with open(path, "w") as f:
                f.write("not json")
            assert read_clip_json(tmpdir) is None


class TestInOutRangeStorage:
    def test_save_load_with_clip_json(self):
        """In/out range saved to clip.json when clip.json exists."""
        from backend.clip_state import InOutRange
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a clip.json first
            write_clip_json(tmpdir, {"source": {"filename": "test.mp4"}})
            rng = InOutRange(in_point=5, out_point=20)
            save_in_out_range(tmpdir, rng)

            loaded = load_in_out_range(tmpdir)
            assert loaded is not None
            assert loaded.in_point == 5
            assert loaded.out_point == 20

            # Verify it's in clip.json, not project.json
            clip_data = read_clip_json(tmpdir)
            assert "in_out_range" in clip_data

    def test_save_load_with_project_json_v1(self):
        """In/out range falls back to project.json for v1 projects."""
        from backend.clip_state import InOutRange
        with tempfile.TemporaryDirectory() as tmpdir:
            # v1: only project.json, no clip.json
            write_project_json(tmpdir, {"version": 1})
            rng = InOutRange(in_point=10, out_point=30)
            save_in_out_range(tmpdir, rng)

            loaded = load_in_out_range(tmpdir)
            assert loaded is not None
            assert loaded.in_point == 10
            assert loaded.out_point == 30

    def test_clear_in_out_range(self):
        from backend.clip_state import InOutRange
        with tempfile.TemporaryDirectory() as tmpdir:
            write_clip_json(tmpdir, {"source": {"filename": "test.mp4"}})
            save_in_out_range(tmpdir, InOutRange(in_point=0, out_point=10))
            save_in_out_range(tmpdir, None)  # clear
            assert load_in_out_range(tmpdir) is None


class TestCreateProject:
    def test_creates_v2_structure(self):
        """Single video creates v2 project with clips/ subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video = os.path.join(tmpdir, "Woman_Jumps.mp4")
            with open(video, "wb") as f:
                f.write(b"\x00" * 100)

            with patch("backend.project.projects_root", return_value=tmpdir):
                project_dir = create_project(video)

            assert os.path.isdir(project_dir)
            assert is_v2_project(project_dir)
            assert os.path.isdir(os.path.join(project_dir, "clips"))

            # Clip subfolder created
            clip_dirs = get_clip_dirs(project_dir)
            assert len(clip_dirs) == 1
            clip_dir = clip_dirs[0]
            assert os.path.isdir(os.path.join(clip_dir, "Source"))

            # Video copied into clip subfolder
            source_files = os.listdir(os.path.join(clip_dir, "Source"))
            assert "Woman_Jumps.mp4" in source_files

            # clip.json written per clip
            clip_data = read_clip_json(clip_dir)
            assert clip_data is not None
            assert clip_data["source"]["filename"] == "Woman_Jumps.mp4"

            # project.json written at project level (v2)
            proj_data = read_project_json(project_dir)
            assert proj_data is not None
            assert proj_data["version"] == 2
            assert "Woman" in proj_data["display_name"]
            assert "clips" in proj_data

    def test_multi_video_creates_one_project(self):
        """Multiple videos create ONE project with multiple clips."""
        with tempfile.TemporaryDirectory() as tmpdir:
            videos = []
            for name in ["Woman_Jumps.mp4", "Man_Walks.mp4", "Dog_Runs.mp4"]:
                path = os.path.join(tmpdir, name)
                with open(path, "wb") as f:
                    f.write(b"\x00" * 100)
                videos.append(path)

            with patch("backend.project.projects_root", return_value=tmpdir):
                project_dir = create_project(videos)

            clip_dirs = get_clip_dirs(project_dir)
            assert len(clip_dirs) == 3

            # Each clip dir has a source video (name matches clip stem)
            clip_source_files = set()
            for cdir in clip_dirs:
                source_dir = os.path.join(cdir, "Source")
                files = os.listdir(source_dir)
                assert len(files) == 1
                clip_source_files.add(files[0])
            assert clip_source_files == {"Woman_Jumps.mp4", "Man_Walks.mp4", "Dog_Runs.mp4"}

            # project.json lists all clips
            proj_data = read_project_json(project_dir)
            assert len(proj_data["clips"]) == 3

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

            # Should get different project folders
            assert project_dir1 != project_dir2
            # Both should have clip subfolders with source videos
            clips1 = get_clip_dirs(project_dir1)
            clips2 = get_clip_dirs(project_dir2)
            assert os.path.isfile(os.path.join(clips1[0], "Source", "test.mp4"))
            assert os.path.isfile(os.path.join(clips2[0], "Source", "test.mp4"))

    def test_duplicate_clip_names_deduplicated(self):
        """Same filename imported twice in one project gets deduped clip names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video = os.path.join(tmpdir, "test.mp4")
            with open(video, "wb") as f:
                f.write(b"\x00" * 100)

            with patch("backend.project.projects_root", return_value=tmpdir):
                project_dir = create_project([video, video])

            clip_dirs = get_clip_dirs(project_dir)
            assert len(clip_dirs) == 2
            names = [os.path.basename(d) for d in clip_dirs]
            assert len(set(names)) == 2  # no duplicates

    def test_custom_display_name(self):
        """display_name sets project.json name and folder stem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            videos = []
            for name in ["clip_a.mp4", "clip_b.mp4"]:
                path = os.path.join(tmpdir, name)
                with open(path, "wb") as f:
                    f.write(b"\x00" * 100)
                videos.append(path)

            with patch("backend.project.projects_root", return_value=tmpdir):
                project_dir = create_project(
                    videos, display_name="My Cool Project",
                )

            # Folder name uses sanitized display_name
            folder = os.path.basename(project_dir)
            assert "My_Cool_Project" in folder

            # project.json stores the original display_name
            proj_data = read_project_json(project_dir)
            assert proj_data["display_name"] == "My Cool Project"

    def test_no_copy_source(self):
        """copy_source=False stores reference without copying."""
        with tempfile.TemporaryDirectory() as tmpdir:
            video = os.path.join(tmpdir, "test.mp4")
            with open(video, "wb") as f:
                f.write(b"\x00" * 100)

            with patch("backend.project.projects_root", return_value=tmpdir):
                project_dir = create_project(video, copy_source=False)

            clip_dir = get_clip_dirs(project_dir)[0]
            source_dir = os.path.join(clip_dir, "Source")
            # Source/ dir exists but is empty (no copy)
            assert os.path.isdir(source_dir)
            assert len(os.listdir(source_dir)) == 0

            # clip.json records copied=False
            clip_data = read_clip_json(clip_dir)
            assert clip_data["source"]["copied"] is False


class TestAddClipsToProject:
    def test_add_clips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video1 = os.path.join(tmpdir, "initial.mp4")
            video2 = os.path.join(tmpdir, "added.mp4")
            for v in [video1, video2]:
                with open(v, "wb") as f:
                    f.write(b"\x00" * 100)

            with patch("backend.project.projects_root", return_value=tmpdir):
                project_dir = create_project(video1)

            new_paths = add_clips_to_project(project_dir, [video2])
            assert len(new_paths) == 1
            assert os.path.isdir(new_paths[0])

            # Project now has 2 clips
            all_clips = get_clip_dirs(project_dir)
            assert len(all_clips) == 2

            # project.json updated
            data = read_project_json(project_dir)
            assert len(data["clips"]) == 2


class TestGetClipDirs:
    def test_v2_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clips_dir = os.path.join(tmpdir, "clips")
            os.makedirs(os.path.join(clips_dir, "clip_a"))
            os.makedirs(os.path.join(clips_dir, "clip_b"))
            result = get_clip_dirs(tmpdir)
            assert len(result) == 2

    def test_v1_project_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # No clips/ dir → treat as single clip
            result = get_clip_dirs(tmpdir)
            assert result == [tmpdir]

    def test_skips_hidden_and_underscore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            clips_dir = os.path.join(tmpdir, "clips")
            os.makedirs(os.path.join(clips_dir, "clip_a"))
            os.makedirs(os.path.join(clips_dir, ".hidden"))
            os.makedirs(os.path.join(clips_dir, "_internal"))
            result = get_clip_dirs(tmpdir)
            assert len(result) == 1
