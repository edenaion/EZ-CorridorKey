"""Tests for natural sort key — handles mixed stems, non-zero-padded numbers."""
import pytest
from ui.preview.natural_sort import natural_sort_key, natsorted


class TestNaturalSortKey:
    def test_numeric_ordering(self):
        result = natsorted(['frame_1', 'frame_10', 'frame_2'])
        assert result == ['frame_1', 'frame_2', 'frame_10']

    def test_zero_padded(self):
        result = natsorted(['shot_0001.exr', 'shot_0010.exr', 'shot_0002.exr'])
        assert result == ['shot_0001.exr', 'shot_0002.exr', 'shot_0010.exr']

    def test_mixed_stems(self):
        """Codex test case: mixed stem formats."""
        result = natsorted(['frame_1', 'frame_2', 'frame_10', 'shot_v2_001'])
        assert result == ['frame_1', 'frame_2', 'frame_10', 'shot_v2_001']

    def test_case_insensitive(self):
        result = natsorted(['Frame_1', 'frame_10', 'FRAME_2'])
        assert result == ['Frame_1', 'FRAME_2', 'frame_10']

    def test_empty_list(self):
        assert natsorted([]) == []

    def test_single_element(self):
        assert natsorted(['file']) == ['file']

    def test_pure_numbers(self):
        result = natsorted(['3', '1', '20', '2'])
        assert result == ['1', '2', '3', '20']

    def test_complex_stems(self):
        """VFX naming: plate_v2_fg_0001.exr"""
        files = [
            'plate_v2_fg_0001.exr',
            'plate_v2_fg_0010.exr',
            'plate_v2_fg_0002.exr',
            'plate_v1_fg_0001.exr',
        ]
        result = natsorted(files)
        assert result == [
            'plate_v1_fg_0001.exr',
            'plate_v2_fg_0001.exr',
            'plate_v2_fg_0002.exr',
            'plate_v2_fg_0010.exr',
        ]
