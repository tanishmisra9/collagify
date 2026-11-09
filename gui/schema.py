"""Declarative description of every adjustable setting.

The browser renders its inspector from this list (via /api/schema), so adding a
control means adding one entry here. Defaults come from collage.py's argparse
parser, so the CLI and GUI cannot drift apart.

`stage` says what a change invalidates:
  analysis : photos must be re-analyzed (slow: color grade, cutouts, faces)
  plan     : layout must be re-planned (fast)
  output   : only affects export
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import collage  # noqa: E402

_D = collage.build_parser().parse_args([])

RATIOS = ["9:16", "4:5", "3:4", "1:1", "2:3", "16:9"]

SETTINGS = [
    # key, label, type, group, stage, extras
    dict(key="ratio", label="Aspect ratio", type="select", group="Canvas", stage="plan",
         options=RATIOS, default=_D.ratio, help="Shape of the canvas, width:height."),
    dict(key="width", label="Width (px)", type="slider", group="Canvas", stage="plan",
         min=540, max=2160, step=60, default=_D.width, help="Export resolution; height follows the ratio."),
    dict(key="seed", label="Seed", type="number", group="Layout", stage="plan",
         min=0, max=999999, default=None, help="Same seed and settings give the same layout. Shuffle picks a new one."),
    dict(key="overlap", label="Overlap", type="slider", group="Layout", stage="plan",
         min=1.0, max=1.5, step=0.01, default=_D.overlap, help="How far photos oversize their cell. 1.0 = no overlap."),
    dict(key="scatter", label="Scatter", type="slider", group="Layout", stage="plan",
         min=0.0, max=1.0, step=0.01, default=_D.scatter, help="How far photos drift off the grid."),
    dict(key="hero_boost", label="Hero boost", type="slider", group="Layout", stage="plan",
         min=1.0, max=3.0, step=0.1, default=_D.hero_boost, help="Extra room for single-photo branches."),
    dict(key="color_strength", label="Color match", type="slider", group="Look", stage="analysis",
         min=0.0, max=1.0, step=0.05, default=_D.color_strength, help="Grade photos toward one shared look. 0 = off."),
    dict(key="cutouts", label="Subject cutouts", type="toggle", group="Subjects", stage="analysis",
         default=not _D.no_cutout, help="AI cutouts enable pops and keep overlaps off subjects."),
    dict(key="faces", label="Face-aware framing", type="toggle", group="Subjects", stage="analysis",
         default=not _D.no_faces, help="Keep every face in frame. Pops only lift people, so off disables pops."),
    dict(key="pop", label="Max pops", type="slider", group="Subjects", stage="plan",
         min=0, max=6, step=1, default=_D.pop, help="Most subjects lifted on top. Quality-gated, so often fewer."),
    dict(key="pop_threshold", label="Pop selectivity", type="slider", group="Subjects", stage="plan",
         min=0.0, max=1.0, step=0.01, default=_D.pop_threshold, help="Higher = only clear standouts pop."),
    dict(key="model", label="Cutout model", type="select", group="Advanced", stage="analysis",
         options=["birefnet-general-lite", "birefnet-general", "u2net", "isnet-general-use"],
         default=_D.model, help="Larger models cut cleaner but download and run slower."),
    dict(key="order", label="Photo order", type="select", group="Advanced", stage="plan",
         options=["random", "name", "date"], default=_D.order,
         help="Mostly breaks ties; layout re-sorts by aspect. Also picks the blurred backdrop photo."),
    dict(key="quality", label="JPEG quality", type="slider", group="Output", stage="output",
         min=60, max=100, step=1, default=_D.quality, help="Export only."),
]

_BY_KEY = {s["key"]: s for s in SETTINGS}
ANALYSIS_KEYS = [s["key"] for s in SETTINGS if s["stage"] == "analysis"]


def defaults():
    return {s["key"]: s["default"] for s in SETTINGS}


def normalize(raw):
    """Validate and clamp a settings dict from the browser. Unknown keys are
    ignored, missing keys take defaults, bad values raise ValueError."""
    out = defaults()
    for key, spec in _BY_KEY.items():
        if key not in raw or spec["type"] == "number":
            continue
        v = raw[key]
        if spec["type"] == "toggle":
            out[key] = bool(v)
        elif spec["type"] == "select":
            if v not in spec["options"]:
                raise ValueError(f"{spec['label']}: '{v}' is not one of {spec['options']}")
            out[key] = v
        else:
            try:
                n = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"{spec['label']}: '{v}' is not a number")
            if n != n:  # NaN
                raise ValueError(f"{spec['label']}: not a number")
            n = max(spec["min"], min(spec["max"], n))
            out[key] = int(round(n)) if isinstance(spec["default"], int) else n
    seed = raw.get("seed")
    try:
        out["seed"] = None if seed in (None, "") else max(0, min(999999, int(seed)))
    except (TypeError, ValueError):
        raise ValueError("Seed must be a whole number.")
    return out
