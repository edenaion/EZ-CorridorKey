"""Probe-driven colour-space filter chain for EXR extraction."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Values ffprobe reports that mean "unknown / missing"
_UNKNOWN_COLOR = {"", "unknown", "unspecified", "reserved", "unknown/unknown", None}


def _is_rgb_pix_fmt(pix_fmt: str) -> bool:
    """Return True if the pixel format is already RGB-family."""
    if not pix_fmt:
        return False
    pf = pix_fmt.lower()
    return (pf.startswith("rgb") or pf.startswith("bgr") or
            pf.startswith("gbr") or pf.startswith("argb") or
            pf.startswith("abgr") or pf == "pal8")


def _is_yuv_pix_fmt(pix_fmt: str) -> bool:
    """Return True for common YUV-family formats."""
    if not pix_fmt:
        return False
    pf = pix_fmt.lower()
    return (pf.startswith("yuv") or pf.startswith("yuva") or
            pf.startswith("nv12") or pf.startswith("nv16") or
            pf.startswith("nv21") or pf.startswith("p010") or
            pf.startswith("p016") or pf.startswith("p210") or
            pf.startswith("p216") or pf.startswith("p410") or
            pf.startswith("p416") or pf.startswith("y210") or
            pf.startswith("y212") or pf.startswith("y216"))


def _clean_color_value(value: str | None) -> str:
    """Normalize ffprobe color values for filter construction."""
    if value is None:
        return ""
    cleaned = str(value).strip().lower()
    return "" if cleaned in _UNKNOWN_COLOR else cleaned


def _default_matrix(width: int, height: int, primaries: str) -> str:
    """Best-effort matrix fallback when metadata is missing."""
    if primaries == "bt2020":
        return "bt2020nc"
    if height == 576:
        return "bt470bg"
    if height in (480, 486):
        return "smpte170m"
    hd = width >= 1280 or height > 576
    return "bt709" if hd else "smpte170m"


def _default_primaries(width: int, height: int, matrix: str) -> str:
    """Best-effort primaries fallback when metadata is missing."""
    if matrix in {"bt2020nc", "bt2020c", "bt2020cl"}:
        return "bt2020"
    if matrix == "bt470bg":
        return "bt470bg"
    if matrix == "smpte170m":
        return "smpte170m"
    hd = width >= 1280 or height > 576
    return "bt709" if hd else "smpte170m"


def _default_transfer(primaries: str, bits_per_raw_sample: int) -> str:
    """Best-effort transfer fallback when metadata is missing."""
    if primaries == "bt2020":
        return "bt2020-12" if bits_per_raw_sample > 10 else "bt2020-10"
    if primaries == "bt470bg":
        return "bt470bg"
    if primaries == "smpte170m":
        return "smpte170m"
    return "bt709"


def _default_range(pix_fmt: str) -> str:
    """Default range fallback for missing ffprobe metadata."""
    pf = (pix_fmt or "").lower()
    return "pc" if pf.startswith("yuvj") else "tv"


# ---------------------------------------------------------------------------
#  ffprobe -> FFmpeg scale filter value mapping
# ---------------------------------------------------------------------------
# ffprobe reports ITU-T H.264/H.265 colour identifiers.  FFmpeg's `scale`
# filter only accepts a subset of named constants.  Values not in the
# "known safe" set get mapped to compatible equivalents.
#
# Built from libavutil/pixfmt.h + libswscale colour table.  If ffprobe
# reports a value we haven't seen, _safe_scale_value() logs a WARNING
# so we can add it before it crashes an extraction.
# ---------------------------------------------------------------------------

# in_color_matrix (swscale colorspace table)
_SCALE_MATRIX_MAP = {
    "bt470bg": "bt601",          # BT.470 System B/G = same matrix as BT.601
    "bt2020c": "bt2020ncl",     # constant-luminance -> non-constant (swscale compat)
}
_KNOWN_MATRICES = {
    "bt709", "fcc", "bt601", "smpte170m", "smpte240m",
    "bt2020nc", "bt2020ncl",
}

# in_primaries
_SCALE_PRIMARIES_MAP = {
    # Most ffprobe primaries pass through. Guard edge cases.
    "film": "bt470m",           # "film" (SMPTE-C) -> bt470m
}
_KNOWN_PRIMARIES = {
    "bt709", "bt470m", "bt470bg", "smpte170m", "smpte240m",
    "film", "bt2020", "smpte428", "smpte431", "smpte432",
}

# in_transfer
_SCALE_TRANSFER_MAP = {
    "bt470bg": "gamma28",       # BT.470 System B/G = gamma 2.8
    "bt470m": "gamma22",        # BT.470 System M = gamma 2.2
    "bt2020-12": "bt2020-12",   # pass through (explicit for clarity)
    "bt2020-10": "bt2020-10",
}
_KNOWN_TRANSFERS = {
    "bt709", "gamma22", "gamma28", "smpte170m", "smpte240m",
    "linear", "log", "log_sqrt", "iec61966-2-4", "bt1361e",
    "iec61966-2-1", "bt2020-10", "bt2020-12", "smpte2084",
    "smpte428", "arib-std-b67",
}


def _safe_scale_value(value: str, mapping: dict, known: set, param_name: str) -> str:
    """Map an ffprobe colour identifier to an FFmpeg scale-filter-safe name.

    Logs a WARNING for unrecognised values so we can add them to the
    mapping before they crash an extraction.
    """
    mapped = mapping.get(value, value)
    if mapped and mapped not in known:
        logger.warning(
            "Unknown %s value '%s' (mapped from '%s') — FFmpeg may reject this. "
            "Add it to _SCALE_%s_MAP in backend/ffmpeg_tools/color.py",
            param_name, mapped, value, param_name.upper(),
        )
    return mapped


def build_exr_vf(video_info: dict) -> str:
    """Build the -vf string for converting to gbrpf32le (EXR output).

    For RGB inputs, just do format conversion.
    For YUV inputs, use an explicit scale+format chain and provide
    input colour metadata directly to the scaler. This preserves the
    current swscale conversion path for well-tagged files and only
    falls back to heuristics when the source metadata is missing.
    """
    pix_fmt = (video_info.get("pix_fmt", "") or "").lower()

    if _is_rgb_pix_fmt(pix_fmt):
        return "format=gbrpf32le"

    # Unknown / oddball formats keep the legacy implicit path.
    if not _is_yuv_pix_fmt(pix_fmt):
        return "format=gbrpf32le"

    cs = _clean_color_value(video_info.get("color_space"))
    cp = _clean_color_value(video_info.get("color_primaries"))
    ct = _clean_color_value(video_info.get("color_transfer"))
    cr = _clean_color_value(video_info.get("color_range"))
    w = video_info.get("width", 0)
    h = video_info.get("height", 0)
    bits = int(video_info.get("bits_per_raw_sample", 0) or 0)

    # ffprobe sometimes writes color_space=gbr for YUV ProRes clips.
    # "gbr" is a pixel-format family name, not a valid YUV matrix tag.
    # FFmpeg's scale filter rejects it with "Unable to parse in_color_matrix
    # option value 'gbr'".  Treat it as missing and fall back to heuristics.
    if cs == "gbr" and _is_yuv_pix_fmt(pix_fmt):
        logger.warning(
            "Ignoring bogus color_space='gbr' on YUV stream (pix_fmt=%s); "
            "falling back to resolution-based matrix heuristic.",
            pix_fmt,
        )
        cs = ""

    if not cs:
        cs = _default_matrix(w, h, cp)
    if not cp:
        cp = _default_primaries(w, h, cs)
    if not ct:
        ct = _default_transfer(cp, bits)
    if not cr:
        cr = _default_range(pix_fmt)

    # Map ffprobe identifiers -> scale-filter-safe names.
    cs = _safe_scale_value(cs, _SCALE_MATRIX_MAP, _KNOWN_MATRICES, "matrix")

    logger.info(
        "EXR colour conversion: pix_fmt=%s matrix=%s range=%s",
        pix_fmt, cs, cr,
    )

    # Only in_color_matrix and in_range are standard swscale options.
    # in_primaries / in_transfer are NOT supported by FFmpeg's scale filter
    # and cause "Option not found" on standard builds.
    #
    # FFmpeg 8.x swscaler rejects YUV->RGB when TRC is null (bug #11585).
    # Prepend setparams to tag the missing TRC -- metadata only, no pixel math.
    raw_ct = video_info.get("color_transfer")
    prefix = f"setparams=color_trc={ct}," if raw_ct is None else ""

    return (
        f"{prefix}scale=in_color_matrix={cs}:in_range={cr},format=gbrpf32le"
    )
