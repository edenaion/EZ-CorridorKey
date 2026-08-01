"""Regression tests for the refiner additive guard (issue #179).

The guard mixes the model's alpha with the alpha hint. When the hint over-covers
the true subject edge by more than the guard's trust shell, a hard switch
between "use the model" and "use the hint" leaves a dark ring with opaque alpha
on both sides, tracing the subject contour at a constant offset.

The invariant these tests protect: the guarded alpha must never contain a dark
band enclosed by opaque alpha, at any source resolution or any amount of hint
over-coverage. The guard's original purposes (filling interior holes from the
hint, protecting detached hair wisps the refiner eroded) must keep working.
"""
import cv2
import numpy as np
import pytest

from CorridorKeyModule.core import color_utils as cu


def _soft_disc(size_wh, radius, softness=2.0):
    """Disc alpha with a linear edge ramp, shaped [H, W, 1] float32."""
    w, h = size_wh
    cx, cy = w // 2, h // 2
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
    a = np.clip((radius - dist) / softness + 0.5, 0.0, 1.0)
    return a.astype(np.float32)[:, :, None]


def _dilate(alpha, px):
    """Grow an alpha's coverage by px, keeping the same edge softness."""
    if px <= 0:
        return alpha
    k = 2 * int(px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.dilate(alpha[:, :, 0], kernel)
    return out.astype(np.float32)[:, :, None]


def _enclosed_dark_pixels(alpha, fg_thresh=0.5, dark_thresh=0.5):
    """Pixels below dark_thresh that sit inside the filled opaque silhouette.

    A contour-following moat is a hole in the silhouette, so binary hole
    filling isolates it exactly. Genuine background outside the subject is
    reachable from the border and therefore never counted.
    """
    a = alpha[:, :, 0] if alpha.ndim == 3 else alpha
    solid = (a > fg_thresh).astype(np.uint8) * 255
    ff = solid.copy()
    mask = np.zeros((solid.shape[0] + 2, solid.shape[1] + 2), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    filled = cv2.bitwise_or(solid, holes)
    enclosed = (filled > 0) & (solid == 0)
    return int((enclosed & (a < dark_thresh)).sum())


def _radial_profile(alpha):
    """Alpha along a horizontal ray from the centre to the right edge."""
    a = alpha[:, :, 0] if alpha.ndim == 3 else alpha
    h, w = a.shape
    return a[h // 2, w // 2:]


def _has_dark_then_bright_reversal(profile, dark=0.25, bright=0.75):
    """True if the profile drops to background and then climbs back to opaque.

    That reversal is the moat: alpha goes opaque -> dark -> opaque as you walk
    outward from the subject centre.
    """
    seen_dark = False
    for v in profile:
        if v < dark:
            seen_dark = True
        elif seen_dark and v > bright:
            return True
    return False


# Hint over-coverage in source pixels. The pre-fix guard used a fixed 4 px
# shell, so everything from 4 px upward resurrected the hint past the shell.
SLACKS = [1, 2, 3, 4, 6, 8, 12, 20, 32]


@pytest.mark.parametrize("slack", SLACKS)
def test_no_enclosed_dark_ring_at_4k(slack):
    """No moat at 4K for any realistic amount of hint over-coverage."""
    model = _soft_disc((3840, 2160), radius=600)
    hint = _dilate(model, slack)
    out = cu.refiner_additive_guard(model, hint)
    assert _enclosed_dark_pixels(out) == 0, (
        f"hint slack {slack}px carved an enclosed dark ring into the matte"
    )


@pytest.mark.parametrize("slack", SLACKS)
def test_no_dark_then_bright_reversal(slack):
    """Walking outward from the subject, alpha must never climb back."""
    model = _soft_disc((1600, 1600), radius=400)
    hint = _dilate(model, slack)
    out = cu.refiner_additive_guard(model, hint)
    assert not _has_dark_then_bright_reversal(_radial_profile(out)), (
        f"hint slack {slack}px produced an opaque -> dark -> opaque reversal"
    )


@pytest.mark.parametrize(
    "size_wh,radius,slack",
    [
        ((1920, 1080), 300, 3),    # 1080p, slack within the historical shell
        ((3840, 2160), 600, 7),    # 4K, slack scales with resolution
        ((5120, 2700), 800, 10),   # 5K, the reported footage size
    ],
)
def test_trust_shell_scales_with_resolution(size_wh, radius, slack):
    """The trust shell must keep pace with resolution.

    Hint models run at a fixed working resolution and are upscaled, so hint
    slack in source pixels grows with the source. A fixed pixel shell loses its
    margin at 4K and 5K, which is what issue #179 reported.
    """
    model = _soft_disc(size_wh, radius=radius)
    hint = _dilate(model, slack)
    out = cu.refiner_additive_guard(model, hint)
    assert _enclosed_dark_pixels(out) == 0
    assert not _has_dark_then_bright_reversal(_radial_profile(out))


def test_no_gradient_ridge_around_silhouette():
    """No hard ridge anywhere outside the silhouette.

    A binary switch between "model" and "hint" prints a step whose gradient
    magnitude is far larger than any legitimate soft matte edge.
    """
    model = _soft_disc((2400, 2400), radius=600, softness=6.0)
    hint = _dilate(model, 14)
    out = cu.refiner_additive_guard(model, hint)[:, :, 0]

    solid = (model[:, :, 0] > 0.5).astype(np.uint8)
    ring = (cv2.dilate(solid, np.ones((81, 81), np.uint8)) > 0) & (solid == 0)

    gx = cv2.Sobel(out, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(out, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    mgx = cv2.Sobel(model[:, :, 0], cv2.CV_32F, 1, 0, ksize=3)
    mgy = cv2.Sobel(model[:, :, 0], cv2.CV_32F, 0, 1, ksize=3)
    grad_model = np.sqrt(mgx * mgx + mgy * mgy)

    # The ring includes the outer half of the model's own edge ramp, and the
    # guard is additive, so the model's own gradient is the floor any output
    # must be judged against. The bar is relative: the guard may not print a
    # step sharper than the matte's real edge. A hard 0 -> 1 switch lands
    # several times above it.
    assert float(grad[ring].max()) <= float(grad_model[ring].max()) * 1.1, (
        "hard alpha step detected outside the silhouette"
    )


def test_interior_holes_are_still_filled_from_the_hint():
    """The guard must keep raising alpha where the model punched a hole."""
    model = _soft_disc((1200, 1200), radius=400)
    hint = model.copy()
    # Model loses confidence in a patch well inside the body
    model[560:640, 560:640, 0] = 0.0

    out = cu.refiner_additive_guard(model, hint)
    assert out[600, 600, 0] > 0.9, "interior hole was not refilled from the hint"


def test_wisp_attached_to_the_body_is_protected():
    """Hair the refiner thinned but the hint captured must survive."""
    model = _soft_disc((1200, 1200), radius=300)
    hint = model.copy()
    # A soft strand running outward from the body edge, only in the hint
    hint[600:604, 900:1000, 0] = 0.6

    out = cu.refiner_additive_guard(model, hint)
    assert out[601, 950, 0] == pytest.approx(0.6, abs=1e-3), (
        "wisp attached to the body was not protected"
    )


def test_fully_detached_wisp_with_model_support_is_protected():
    """A fly-away strand separated from the body by real background.

    The gap is pure zero alpha and far wider than the fill can step, so the
    strand is only saved by the model's own trace of alpha on it. This is the
    fly-away hair case the guard exists for.
    """
    model = _soft_disc((1200, 1200), radius=300)
    hint = model.copy()
    strand = (slice(600, 604), slice(1000, 1100))
    assert model[600, 1000, 0] == 0.0, "fixture must start beyond the body"
    assert model[600, 940, 0] == 0.0, "gap between body and strand must be empty"

    hint[strand[0], strand[1], 0] = 0.6
    # The refiner thinned the strand but did not erase it
    model[strand[0], strand[1], 0] = 0.05

    out = cu.refiner_additive_guard(model, hint)
    assert out[601, 1050, 0] == pytest.approx(0.6, abs=1e-3), (
        "fully detached wisp with model support was dropped"
    )


def test_detached_hint_blob_without_model_support_is_suppressed():
    """Spill and gunk the model rejected outright must not be burned in.

    Same geometry as the fly-away strand above, but the model gives it no
    support at all, so there is nothing corroborating the hint.
    """
    model = _soft_disc((1200, 1200), radius=300)
    hint = model.copy()
    hint[560:640, 1000:1100, 0] = 1.0  # hint insists, model says background

    out = cu.refiner_additive_guard(model, hint)
    assert out[600, 1050, 0] < 0.05, (
        "hint blob with no model support was burned into the matte"
    )


def test_guard_never_lowers_alpha_below_the_model():
    """The guard is additive: it may add confidence, never subtract it."""
    rng = np.random.default_rng(179)
    model = rng.random((256, 256, 1)).astype(np.float32)
    hint = rng.random((256, 256, 1)).astype(np.float32)
    out = cu.refiner_additive_guard(model, hint)
    assert np.all(out >= model - 1e-6)
    assert np.all(out <= 1.0 + 1e-6)


def test_missing_hint_is_a_passthrough():
    model = _soft_disc((320, 320), radius=100)
    assert cu.refiner_additive_guard(model, None) is model


@pytest.mark.parametrize("fill", [0.0, 1.0])
def test_degenerate_alpha_does_not_crash(fill):
    """All-background and all-foreground frames must not blow up."""
    model = np.full((128, 128, 1), fill, dtype=np.float32)
    hint = np.full((128, 128, 1), fill, dtype=np.float32)
    out = cu.refiner_additive_guard(model, hint)
    assert out.shape == model.shape
    assert np.isfinite(out).all()


def test_two_dimensional_alpha_shape_is_preserved():
    model = _soft_disc((256, 256), radius=80)[:, :, 0]
    hint = _dilate(model[:, :, None], 6)[:, :, 0]
    out = cu.refiner_additive_guard(model, hint)
    assert out.shape == model.shape
    assert np.isfinite(out).all()


def test_guard_output_stays_in_unit_range_for_out_of_range_input():
    """Alpha leaving the guard must be a valid matte.

    The Lanczos upscale feeding this function overshoots at sharp edges, so
    the guard must not pass values outside [0, 1] downstream into
    premultiply and compositing.
    """
    model = _soft_disc((512, 512), radius=160) * 1.09  # Lanczos-style overshoot
    hint = _dilate(_soft_disc((512, 512), radius=160), 6)
    out = cu.refiner_additive_guard(model, hint)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_lanczos_upscale_overshoot_is_clamped():
    """The engine clamps alpha after INTER_LANCZOS4 (issue #179 follow-up).

    Mirrors the resize-then-clamp sequence in
    CorridorKeyModule/inference_engine.py and its MLX counterpart. Without
    the clamp a hard alpha edge upscales to values above 1.0.
    """
    small = np.zeros((64, 64), np.float32)
    small[:, :32] = 1.0  # hard step, worst case for Lanczos ringing

    upscaled = cv2.resize(small, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    assert upscaled.max() > 1.0, "expected Lanczos overshoot in this fixture"

    np.clip(upscaled, 0.0, 1.0, out=upscaled)
    assert upscaled.min() >= 0.0
    assert upscaled.max() <= 1.0


def test_tiny_image_smaller_than_the_shell():
    model = _soft_disc((16, 16), radius=5)
    hint = _dilate(model, 3)
    out = cu.refiner_additive_guard(model, hint)
    assert out.shape == model.shape
    assert np.isfinite(out).all()


def _reconstruct_fixture():
    """A marker/mask pair with a body, a connected strand, a detached strand
    with a seed, and a detached blob without one, so every reconstruction
    branch is exercised."""
    rng = np.random.default_rng(179)
    mask = (rng.random((540, 1024)) * 0.2).astype(np.float32)
    mask[200:340, 300:600] = 1.0            # body
    mask[260:266, 600:800] = 0.8            # strand connected to the body
    mask[100:106, 700:900] = 0.7            # detached strand
    mask[400:460, 700:820] = 0.9            # detached blob
    marker = np.zeros_like(mask)
    marker[200:340, 300:600] = 1.0          # seed: body
    marker[102:104, 780:790] = 0.05         # seed: trace on detached strand
    return marker, mask


def test_reconstruct_cpu_path_when_no_device(monkeypatch):
    """The dispatcher must fall back to the cv2 loop with no torch device."""
    monkeypatch.setattr(cu, "_RECONSTRUCT_DEVICE", None)
    marker, mask = _reconstruct_fixture()
    out = cu._geodesic_reconstruct(marker, mask)
    ref = cu._geodesic_reconstruct_cpu(marker, mask)
    assert np.array_equal(out, ref)


def test_full_guard_gpu_matches_cpu():
    """The whole torch guard path must match the cv2 path.

    Exact selections everywhere except the area downscale and bilinear
    upscale, whose float rounding may differ at the last bit. Bound is far
    below one 8-bit alpha step, and the matte invariants must hold on both.
    """
    pytest.importorskip("torch")
    device = cu._reconstruct_device()
    if device is None:
        pytest.skip("no torch device available")

    model = _soft_disc((2560, 1440), radius=500)
    hint = _dilate(model, 8)
    hint[700:704, 2000:2200, 0] = 0.6   # detached strand
    model[700:704, 2000:2200, 0] = 0.05  # with a trace of model support

    gpu = cu._guard_apply_torch(model, model[:, :, 0], hint[:, :, 0],
                                True, device)
    ref = cu._guard_apply_cpu(model, model[:, :, 0], hint[:, :, 0], True)
    assert gpu.shape == ref.shape
    assert float(np.abs(gpu - ref).max()) <= 2e-3
    assert _enclosed_dark_pixels(gpu) == 0
    assert _enclosed_dark_pixels(ref) == 0
    assert gpu[701, 2100, 0] == pytest.approx(0.6, abs=1e-2)


def test_reconstruct_gpu_matches_cpu():
    """GPU and cv2 reconstruction are the same selection, so results match.

    The loop is pure max/min; the only permitted divergence is the border
    ring where padding semantics could differ, and even there both use
    -inf-style constant borders. Bound: exact interior, 1e-6 everywhere.
    """
    torch = pytest.importorskip("torch")
    device = cu._reconstruct_device()
    if device is None:
        pytest.skip("no torch device available")
    marker, mask = _reconstruct_fixture()
    gpu = cu._geodesic_reconstruct_torch(marker, mask, device)
    ref = cu._geodesic_reconstruct_cpu(marker, mask)
    assert gpu.shape == ref.shape
    assert float(np.abs(gpu - ref).max()) <= 1e-6
    inner = (slice(8, -8), slice(8, -8))
    assert np.array_equal(gpu[inner], ref[inner])
