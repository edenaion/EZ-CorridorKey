import numpy as np
import pytest
import torch

from sam2_tracker.wrapper import PromptFrame, SAM2Tracker


def test_sanitize_prompt_frame_clamps_points_box_and_mask():
    prompt = PromptFrame(
        frame_index=3,
        positive_points=[(-4.2, 8.8), (50.1, 98.9), (50.1, 98.9)],
        negative_points=[(9.9, -3.0)],
        box=(10.0, 5.0, 40.0, 30.0),
        mask=np.array([[0, 1], [255, 0]], dtype=np.uint8),
    )

    sanitized = SAM2Tracker._sanitize_prompt_frame(prompt, (2, 2))

    assert sanitized.frame_index == 3
    assert sanitized.positive_points == []
    assert sanitized.negative_points == []
    assert sanitized.box is None
    assert sanitized.mask is not None
    assert sanitized.mask.dtype == np.uint8
    assert sanitized.mask.tolist() == [[0, 255], [255, 0]]


def test_sanitize_prompt_frame_requires_2d_mask_matching_frame_size():
    prompt = PromptFrame(frame_index=0, mask=np.zeros((4, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="must match the input frame size"):
        SAM2Tracker._sanitize_prompt_frame(prompt, (5, 4))

    with pytest.raises(ValueError, match="must be 2D"):
        SAM2Tracker._sanitize_prompt_frame(
            PromptFrame(frame_index=0, mask=np.zeros((4, 4, 2), dtype=np.uint8)),
            (4, 4),
        )


def test_has_foreground_signal_accepts_nonempty_mask_or_positive_points():
    assert SAM2Tracker._has_foreground_signal(
        PromptFrame(frame_index=0, mask=np.array([[0, 255]], dtype=np.uint8))
    )
    assert SAM2Tracker._has_foreground_signal(
        PromptFrame(frame_index=0, positive_points=[(1.0, 1.0)])
    )
    assert not SAM2Tracker._has_foreground_signal(
        PromptFrame(frame_index=0, mask=np.zeros((2, 2), dtype=np.uint8))
    )


def test_iter_prompt_refinement_batches_preserves_points_and_boxes():
    prompt = PromptFrame(
        frame_index=0,
        positive_points=[(float(i), float(i)) for i in range(24)],
        negative_points=[(100.0 + i, 200.0 + i) for i in range(8)],
        box=(10.0, 20.0, 30.0, 40.0),
    )

    batches = list(SAM2Tracker._iter_prompt_refinement_batches(prompt))

    assert len(batches) == 4
    first_points, first_labels, first_box, first_clear = batches[0]
    assert first_clear is True
    assert first_box == prompt.box
    assert first_points.shape == (8, 2)
    assert first_labels.tolist() == [1, 1, 1, 1, 1, 1, 0, 0]

    for batch_points, batch_labels, batch_box, clear_old_points in batches[1:]:
        assert clear_old_points is False
        assert batch_box is None
        assert batch_points.shape == (8, 2)
        assert batch_labels.tolist() == [1, 1, 1, 1, 1, 1, 0, 0]


def test_track_video_refines_sparse_point_batches(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakePredictor:
        def init_state(self, **kwargs):
            return {"video_path": kwargs["video_path"]}

        def add_new_points_or_box(
            self,
            *,
            inference_state,
            frame_idx,
            obj_id,
            points=None,
            labels=None,
            clear_old_points=True,
            box=None,
        ):
            calls.append(
                {
                    "frame_idx": frame_idx,
                    "obj_id": obj_id,
                    "points": None if points is None else points.copy(),
                    "labels": None if labels is None else labels.copy(),
                    "clear_old_points": clear_old_points,
                    "box": box,
                }
            )
            mask_logits = torch.ones((1, 1, 512, 512), dtype=torch.float32)
            return frame_idx, np.array([1], dtype=np.int32), mask_logits

        def propagate_in_video(self, *args, **kwargs):
            return iter(())

    tracker = SAM2Tracker(device="cpu")
    monkeypatch.setattr(tracker, "_get_predictor", lambda **kwargs: FakePredictor())

    prompt = PromptFrame(
        frame_index=0,
        positive_points=[(float(i), float(i)) for i in range(24)],
        negative_points=[(100.0 + i, 200.0 + i) for i in range(8)],
        box=(10.0, 20.0, 30.0, 40.0),
    )
    frames = [np.zeros((512, 512, 3), dtype=np.uint8)]

    masks = tracker.track_video(frames, [prompt])

    assert len(masks) == 1
    assert len(calls) == 4
    assert calls[0]["clear_old_points"] is True
    assert calls[0]["box"] == prompt.box
    assert calls[0]["points"].shape == (8, 2)
    assert calls[0]["labels"].tolist() == [1, 1, 1, 1, 1, 1, 0, 0]
    for call in calls[1:]:
        assert call["clear_old_points"] is False
        assert call["box"] is None
        assert call["points"].shape == (8, 2)
        assert call["labels"].tolist() == [1, 1, 1, 1, 1, 1, 0, 0]
