import json

from backend.annotation_prompts import load_annotation_mask_frames, load_annotation_prompt_frames


def test_load_annotation_prompts_box_expands_by_brush_radius(tmp_path):
    (tmp_path / "annotations.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "points": [[100, 200]],
                        "brush_type": "fg",
                        "radius": 10.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    prompts = load_annotation_prompt_frames(str(tmp_path))

    assert len(prompts) == 1
    assert prompts[0].frame_index == 0
    # Radius expansion should include more than the center point.
    assert (100.0, 200.0) in prompts[0].positive_points
    assert len(prompts[0].positive_points) > 1
    assert prompts[0].negative_points == []
    assert prompts[0].box == (90.0, 190.0, 110.0, 210.0)


def test_load_annotation_prompts_preserves_all_points_by_default(tmp_path):
    points = [[float(i), 50.0] for i in range(12)]
    (tmp_path / "annotations.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "points": points,
                        "brush_type": "fg",
                        "radius": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    prompts = load_annotation_prompt_frames(str(tmp_path))

    assert len(prompts) == 1
    assert len(prompts[0].positive_points) == 12
    assert prompts[0].negative_points == []


def test_load_annotation_prompts_sampling_remains_opt_in(tmp_path):
    points = [[float(i), 50.0] for i in range(12)]
    (tmp_path / "annotations.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "points": points,
                        "brush_type": "fg",
                        "radius": 0.0,
                    },
                    {
                        "points": points,
                        "brush_type": "bg",
                        "radius": 0.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    prompts = load_annotation_prompt_frames(str(tmp_path), max_points_per_stroke=4)

    assert len(prompts) == 1
    assert len(prompts[0].positive_points) == 4
    assert len(prompts[0].negative_points) == 4


def test_load_annotation_prompts_radius_expansion_is_bounded(tmp_path):
    points = [[float(i), 50.0] for i in range(200)]
    (tmp_path / "annotations.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "points": points,
                        "brush_type": "fg",
                        "radius": 200.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    prompts = load_annotation_prompt_frames(str(tmp_path))

    assert len(prompts) == 1
    # Radius semantics are preserved, but point count stays bounded.
    assert 1 < len(prompts[0].positive_points) <= 1024


def test_load_annotation_prompts_background_strokes_stay_sparse_by_default(tmp_path):
    points = [[float(i), 50.0] for i in range(200)]
    (tmp_path / "annotations.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "points": [[100.0, 100.0], [120.0, 120.0]],
                        "brush_type": "fg",
                        "radius": 25.0,
                    },
                    {
                        "points": points,
                        "brush_type": "bg",
                        "radius": 200.0,
                    },
                    {
                        "points": points,
                        "brush_type": "bg",
                        "radius": 200.0,
                    },
                    {
                        "points": points,
                        "brush_type": "bg",
                        "radius": 200.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    prompts = load_annotation_prompt_frames(str(tmp_path))

    assert len(prompts) == 1
    assert len(prompts[0].positive_points) > 1
    assert 1 <= len(prompts[0].negative_points) <= 24


def test_load_annotation_mask_frames_rasterize_exact_brush_geometry(tmp_path):
    (tmp_path / "annotations.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "points": [[8, 8], [24, 8]],
                        "brush_type": "fg",
                        "radius": 4.0,
                    },
                    {
                        "points": [[24, 8]],
                        "brush_type": "bg",
                        "radius": 2.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    prompts = load_annotation_mask_frames(str(tmp_path), width=32, height=24)

    assert len(prompts) == 1
    mask = prompts[0].mask
    assert mask.shape == (24, 32)
    assert mask[8, 8] == 255
    assert mask[8, 16] == 255
    assert mask[8, 24] == 0


def test_load_annotation_mask_frames_respects_allowed_indices(tmp_path):
    (tmp_path / "annotations.json").write_text(
        json.dumps(
            {
                "0": [
                    {
                        "points": [[8, 8]],
                        "brush_type": "fg",
                        "radius": 4.0,
                    }
                ],
                "3": [
                    {
                        "points": [[16, 16]],
                        "brush_type": "fg",
                        "radius": 4.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    prompts = load_annotation_mask_frames(
        str(tmp_path),
        width=32,
        height=24,
        allowed_indices=[3],
    )

    assert len(prompts) == 1
    assert prompts[0].frame_index == 3
