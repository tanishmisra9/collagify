#!/usr/bin/env python3
"""
collage.py — Generate a layered "sticker board" photo collage from a folder.

================================================================================
WHAT THIS PROGRAM DOES (in plain English)
================================================================================
You point it at a folder of photos. It arranges them into one tall image (a
"collage") where the photos overlap a little like a stack of printed pictures on
a table, and ONE person/subject is cut out from its background and lifted on top
with a soft shadow for a bit of depth. It then saves that combined image.

The hard part is doing this *well*: every person should stay fully visible (never
sliced in half by a crop, never hidden under an overlapping photo), the photos
should fit together without ugly gaps, and the result should look good no matter
which photos you give it. Most of the code below exists to make those guarantees.

================================================================================
THE PIPELINE (the order things happen)
================================================================================
1. Find the image files in the folder.                          -> find_images
2. Open them, fix sideways-phone rotation, put them in order.    -> load_images
3. Color-grade all photos toward one shared look, so a set shot  -> unify_colors
   in different lighting feels like one cohesive piece.
4. Detect faces in each photo, so framing and layout can keep    -> detect_faces
   every face fully in view.
5. Use an AI model (rembg) to "cut out" the subject in each      -> subject_cutout
   photo (separate the person/object from the background).
6. Plan an invisible grid ("skeleton") that gives each photo a   -> layout_cells
   cell shaped to fit its subject/faces.
7. Composite (paste) everything onto one canvas:                 -> build_collage
   - oversize + jitter the cells so photos overlap a little,
   - paint photo backgrounds back-to-front WITHOUT covering any
     subject that's already there,
   - lift ONE subject on top with a shadow ("the pop").
8. Save the finished image to disk.                              -> main

================================================================================
GLOSSARY (terms used all over this file)
================================================================================
- pixel: one tiny colored dot. An image is a grid of pixels.
- width / height: how many pixels across / down. Always in pixels here.
- aspect ratio: width / height. >1 is wide ("landscape"), <1 is tall
  ("portrait"), =1 is square. A phone selfie is usually portrait (~0.75).
- canvas: the big blank image we paste all the photos onto. The final collage.
- Pillow / PIL: the image library we use. `Image` objects hold pictures.
  `Image.LANCZOS` is a high-quality way to resize.
- RGB: a normal color image (Red, Green, Blue channels).
- RGBA: a color image with an extra "Alpha" channel = transparency. Alpha 255
  means fully solid, 0 means fully see-through. A "cutout" is RGBA: the subject
  is solid, the removed background is transparent.
- mask: a black-and-white image (mode "L", values 0-255) used when pasting. White
  (255) = paste this pixel, black (0) = leave what's underneath. It's how we get
  shaped/soft edges instead of always pasting a hard rectangle.
- rembg: the AI library that looks at a photo and separates the foreground
  subject from the background, giving us the cutout described above.
- numpy / np: a library for fast math on big grids of numbers. We use boolean
  numpy arrays (every pixel is True/False) as "maps" — e.g. a map of which
  pixels a subject occupies. `A & B` = overlap of two maps, `~A` = the opposite.
- guillotine layout: a way to divide a rectangle by making one straight cut into
  two pieces, then cutting each piece again, and so on. Like slicing a sheet of
  paper with a paper cutter ("guillotine"). It tiles the canvas with no gaps.
- recursion: a function that calls itself to handle smaller pieces of the same
  problem. plan_layout/realize_layout do this to cut the canvas up.

Usage:
    python collage.py
    python collage.py --seed 42 --scatter 0.3 --overlap 1.2
    python collage.py --no-cutout        # plain overlap only (skip the AI cutouts)
"""

# --- Standard-library imports (come with Python) ---------------------------
import argparse          # parses command-line options like --width 1080
import math              # math.log() — used by the layout cost function
import random            # the seeded random-number generator for variety
import sys               # sys.stderr / sys.exit — for error messages & quitting
from collections import namedtuple  # a tiny lightweight record type (see Cutout)
from pathlib import Path # a friendly way to handle file paths

# --- Third-party imports (installed via requirements.txt) ------------------
import numpy as np                            # fast grid math (the True/False maps)
from PIL import Image, ImageOps, ImageFilter  # Pillow: open/resize/paste images
# (rembg and scipy are imported *inside* the functions that need them, so the
#  program can still run --no-cutout even if rembg isn't installed.)

# File extensions we treat as images. Anything else in the folder is ignored.
# It's a set ({...}) because "is this extension in the set?" is a fast check.
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# How much a "popped" subject is scaled up when lifted on top (1.0 = same size as
# the background subject). A little >1 makes the cutout grow past where the subject
# sits in its photo, so the lift actually reads as depth instead of an invisible
# in-place re-paste. Kept modest so it never swamps neighbors.
POP_LIFT = 1.06

# Most photos a single collage should contain. More than this gets cramped in a
# 9:16 frame, so main() rejects larger inputs (override with --max-photos).
MAX_PHOTOS = 6


def build_parser():
    """Build the argparse parser. Split from parse_args so the GUI can read the
    same option names and defaults (one source of truth).

    Original docstring for the options:

    `argparse` is Python's standard tool for this. Each `add_argument` line below
    defines one option, its default value, and a help string (shown by
    `python collage.py --help`). Returns an object where, e.g., `args.width` is
    the chosen width. We don't need to validate much here — argparse handles
    types (e.g. `type=int`) and rejects unknown options for us.
    """
    p = argparse.ArgumentParser(
        description="Generate a layered, overlapping sticker-board collage from a folder."
    )
    p.add_argument("--input-dir", default="pics",
                    help="Directory containing source images (default: pics)")
    p.add_argument("--output", default="output/collage.jpg",
                    help="Output file path (default: output/collage.jpg)")
    p.add_argument("--width", type=int, default=1080,
                    help="Output canvas width in px (default: 1080)")
    p.add_argument("--ratio", default="9:16",
                    help="Canvas aspect ratio as W:H, e.g. 4:5, 9:16, 3:4 (default: 9:16)")
    p.add_argument("--overlap", type=float, default=1.05,
                    help="How much each photo oversizes its cell so neighbors overlap "
                         "(1.0 = no overlap; default: 1.05)")
    p.add_argument("--scatter", type=float, default=0.06,
                    help="How much photos are jittered off-grid, 0-1 (default: 0.06)")
    p.add_argument("--pop", type=int, default=1,
                    help="Maximum subjects that pop out over neighbors. Quality-gated, "
                         "so it usually pops 0 and only lifts a clear standout "
                         "(0 = never; default: 1)")
    p.add_argument("--pop-threshold", type=float, default=0.38,
                    help="Minimum tastefulness score for a subject to pop, higher = "
                         "more selective (default: 0.38)")
    p.add_argument("--model", default="birefnet-general-lite",
                    help="rembg segmentation model (default: birefnet-general-lite)")
    p.add_argument("--no-cutout", action="store_true",
                    help="Skip rembg subject cutouts entirely; plain overlap only")
    p.add_argument("--color-strength", type=float, default=0.35,
                    help="How strongly photos are graded toward a shared look, 0-1 "
                         "(0 = off, 1 = full match; default: 0.35)")
    p.add_argument("--no-color", action="store_true",
                    help="Skip color cohesion grading entirely")
    p.add_argument("--no-faces", action="store_true",
                    help="Skip face detection; frame on the rembg subject only "
                         "(also disables pops, which only ever lift detected people)")
    p.add_argument("--hero-boost", type=float, default=1.5,
                    help="How much extra space single-image branches get, 1.0 = no boost (default: 1.5)")
    p.add_argument("--order", choices=["random", "name", "date"], default="random",
                    help="Order images are placed in (default: random)")
    p.add_argument("--seed", type=int, default=None,
                    help="Random seed, for a reproducible layout")
    p.add_argument("--recursive", action="store_true",
                    help="Also search subdirectories of input-dir")
    p.add_argument("--max-photos", type=int, default=MAX_PHOTOS,
                    help=f"Reject inputs with more than this many images, since more "
                         f"get cramped in the frame (default: {MAX_PHOTOS})")
    p.add_argument("--quality", type=int, default=100,
                    help="JPEG output quality 1-100 (default: 100)")
    return p


def parse_args():
    """Read the command-line options the user typed and return them as an object."""
    return build_parser().parse_args()


def find_images(input_dir: Path, recursive: bool):
    """Return a sorted list of image file paths inside `input_dir`.

    Why: we need the list of photos to put in the collage.
    How: `input_dir.glob(pattern)` lists files. The pattern "*" means "things
    directly in this folder"; "**/*" means "this folder and all subfolders"
    (used when --recursive is given). We keep only real files whose extension is
    in SUPPORTED_EXTS (lower-cased so ".JPG" still matches). `sorted(...)` gives a
    stable, predictable order (important so runs are reproducible).
    """
    pattern = "**/*" if recursive else "*"
    return sorted(
        f for f in input_dir.glob(pattern)
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    )


def load_images(paths, order, rng):
    """Open each image file and return a list of (path, Pillow image) pairs.

    Why three steps happen to each photo:
      1. `Image.open` reads the file.
      2. `ImageOps.exif_transpose` fixes rotation. Phones store photos with a
         hidden "rotate me" flag (EXIF orientation) instead of actually rotating
         the pixels; without this a portrait photo can come in sideways.
      3. `.convert("RGB")` forces a plain color image. Some files are grayscale,
         palette-based, or have transparency; converting up front means the rest
         of the code can assume every photo is normal RGB.

    Edge case handled: a corrupt/unreadable file would normally crash the program.
    The try/except prints a warning to stderr and SKIPS just that file, so one bad
    photo doesn't ruin the whole run.

    Then we order the photos:
      - "name": alphabetical by filename.
      - "date": oldest-modified first.
      - "random" (default): shuffled. We shuffle with `rng` (the *seeded* random
        generator) rather than plain random, so the same --seed always produces
        the same shuffle — making results reproducible.
    """
    images = []
    for p in paths:
        try:
            img = Image.open(p)
            img = ImageOps.exif_transpose(img)  # respect camera rotation
            img = img.convert("RGB")
            images.append((p, img))
        except Exception as e:
            print(f"  ! skipping {p.name}: {e}", file=sys.stderr)

    return order_images(images, order, rng)


def order_images(images, order, rng):
    """Order (path, image) pairs in place and return them: "name", "date", or
    "random" (shuffled with the seeded `rng`). Split out so the GUI can re-order
    already-analyzed photos without reloading them."""
    if order == "name":
        images.sort(key=lambda t: t[0].name.lower())
    elif order == "date":
        images.sort(key=lambda t: t[0].stat().st_mtime)
    else:  # random
        rng.shuffle(images)

    return images


def parse_ratio(ratio_str):
    """Turn a "W:H" text like "9:16" into a single number (width/height = 0.5625).

    Why: the canvas shape is given as a ratio string on the command line, but the
    math needs a plain number. We split on ":" and divide.

    Edge case: if the user types something invalid (e.g. "16x9" or "abc"), the
    split/float conversion fails; we catch that and raise a clear ValueError that
    main() turns into a friendly error message instead of a raw crash.
    """
    try:
        w, h = ratio_str.split(":")
        return float(w) / float(h)
    except Exception:
        raise ValueError(f"Invalid --ratio '{ratio_str}', expected format like 4:5")


# ===========================================================================
# COLOR COHESION
#
# Photos shot in different lighting (warm indoor vs. cool outdoor) clash and the
# collage looks like a pile of unrelated pictures. unify_colors nudges every
# photo toward ONE shared look so the set feels intentional — like one roll of
# film. We use "Reinhard color transfer," the standard technique: match each
# photo's color statistics (average + spread) to the whole set's average.
# ===========================================================================

def unify_colors(images, strength):
    """Grade every photo toward the set's average color/brightness and return a new
    list of (path, graded image). `strength` is 0-1: 0 returns the photos
    unchanged, 1 fully matches them to the shared target, values in between blend.

    Why LAB color space (not plain RGB): LAB separates lightness (L) from the two
    color axes (a = green<->red, b = blue<->yellow). Matching statistics there
    shifts color and exposure in a way that looks natural to the eye, instead of
    tinting the RGB channels independently (which can produce ugly casts).

    How (Reinhard transfer):
      1. Convert each photo to LAB and record its per-channel mean and standard
         deviation (std = how spread out / contrasty that channel is).
      2. The shared TARGET is the average of all photos' means and stds — the
         "middle" look of the whole set.
      3. For each photo and each channel: subtract the photo's own mean, rescale
         by (target_std / photo_std) so its contrast matches the target, then add
         the target mean. That recenters and rescales the photo onto the shared
         look.
      4. Blend the result with the original by `strength` so the effect stays
         tasteful rather than flattening every photo to identical color.

    Edge cases:
      - strength <= 0, or fewer than 2 photos: nothing to unify, return as-is.
      - A near-flat channel has std ≈ 0 (e.g. a solid-color photo); dividing by it
        would blow up, so we guard with a small floor (1e-5) on the photo's std.
      - We clip back into the valid 0-255 range before converting LAB->RGB, since
        the rescale can push values slightly out of range.
    """
    if strength <= 0 or len(images) < 2:
        return images

    import cv2  # imported here so the rest of the program runs without OpenCV

    # Convert each photo to LAB once, as float32 for accurate math.
    labs = []
    stats = []  # (mean, std) per photo, each a length-3 array over L,a,b
    for _, img in images:
        rgb = np.asarray(img, dtype=np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        labs.append(lab)
        # axis=(0,1) averages over all pixels, leaving one number per channel.
        stats.append((lab.mean(axis=(0, 1)), lab.std(axis=(0, 1))))

    # Shared target = the set's average mean and average spread.
    target_mean = np.mean([m for m, _ in stats], axis=0)
    target_std = np.mean([s for _, s in stats], axis=0)

    out = []
    for (path, img), lab, (mean, std) in zip(images, labs, stats):
        safe_std = np.maximum(std, 1e-5)                 # never divide by ~0
        graded = (lab - mean) * (target_std / safe_std) + target_mean
        # Blend graded vs. original by strength, then clip to the legal LAB range.
        blended = lab * (1 - strength) + graded * strength
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(blended, cv2.COLOR_LAB2RGB)
        out.append((path, Image.fromarray(rgb, "RGB")))
    return out


# ===========================================================================
# FACE DETECTION
#
# rembg tells us where the whole subject is, but not specifically where FACES
# are. detect_faces uses InsightFace's SCRFD model (state-of-the-art accuracy) to
# find every face, so the layout can give group photos wide-enough cells and the
# crop can guarantee faces stay in view. Faces are the thing a viewer cares about
# most, so this is the highest-value signal for framing.
# ===========================================================================

# We build the detector once and reuse it (loading the model is slow). This
# module-level holder caches it after the first call.
_FACE_APP = None


def _face_app():
    """Create (once) and return the InsightFace detector, or None if it can't load.

    Returns None on any failure (package missing, model download blocked, etc.) so
    the caller can simply fall back to subject-based framing — face detection is an
    enhancement, never a hard requirement.
    """
    global _FACE_APP
    if _FACE_APP is None:
        try:
            from insightface.app import FaceAnalysis
            # allowed_modules=['detection'] loads ONLY the face-finder (we don't
            # need landmarks/recognition), which is faster and lighter.
            app = FaceAnalysis(allowed_modules=["detection"],
                               providers=["CPUExecutionProvider"])
            # ctx_id=-1 means CPU; det_size is the internal working resolution.
            app.prepare(ctx_id=-1, det_size=(640, 640))
            _FACE_APP = app
        except Exception as e:
            print(f"  ! face detection unavailable ({e}); framing on subject only",
                  file=sys.stderr)
            _FACE_APP = False  # remember the failure so we don't retry every photo
    return _FACE_APP or None


def face_detector_failed():
    """True if loading the face detector failed earlier in this process."""
    return _FACE_APP is False


def retry_face_detector():
    """Forget a failed detector load so the next detect_faces tries again. A long-lived
    process (the GUI) should not stay without faces just because one launch was offline."""
    global _FACE_APP
    if _FACE_APP is False:
        _FACE_APP = None


def detect_faces(img, min_score=0.5, min_rel_area=0.18):
    """Return a list of the MAIN face boxes in `img`, each as (x0, y0, x1, y1)
    normalized to 0-1 (fractions of the image), or an empty list if none /
    detector unavailable.

    Normalizing to 0-1 makes the boxes independent of the photo's pixel size, so
    they combine cleanly with the rembg subject box (also 0-1) downstream.

    Why BGR: OpenCV and InsightFace expect Blue-Green-Red channel order, but Pillow
    gives Red-Green-Blue, so we reverse the last axis ([:, :, ::-1]).

    Two filters keep us focused on the people the photo is actually ABOUT, not the
    crowd behind them (a real problem: a graduation shot found 9 faces, mostly
    background strangers, which would drag the "keep in frame" box across the whole
    image and frame on the crowd):
      - min_score: drop low-confidence detections (det_score below ~0.5).
      - min_rel_area: drop any face much smaller than the biggest one. Distant
        background faces are tiny; foreground subjects are large. We keep faces
        whose area is at least `min_rel_area` (18%) of the largest face's area, so
        a foreground group stays and the far-away crowd is dropped.

    Edge cases: any detector error returns [] (treated as "no faces" -> we fall
    back to the subject box). Boxes are clamped into [0,1]. A single face is always
    kept (it is its own largest).
    """
    app = _face_app()
    if app is None:
        return []
    try:
        bgr = np.asarray(img.convert("RGB"))[:, :, ::-1]
        iw, ih = img.width, img.height
        # Collect (area, normalized-box) for each confident detection.
        found = []
        for f in app.get(bgr):
            if float(f.det_score) < min_score:
                continue
            x0, y0, x1, y1 = f.bbox
            box = (max(0.0, min(1.0, x0 / iw)), max(0.0, min(1.0, y0 / ih)),
                   max(0.0, min(1.0, x1 / iw)), max(0.0, min(1.0, y1 / ih)))
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
            if area > 0:
                found.append((area, box))
        if not found:
            return []
        # Keep only faces comparable in size to the largest (drop distant crowd).
        biggest = max(a for a, _ in found)
        return [box for a, box in found if a >= biggest * min_rel_area]
    except Exception as e:
        print(f"  ! face detection failed: {e}", file=sys.stderr)
        return []


def _faces_on_subject(faces, cut):
    """Keep only the faces that sit on the rembg foreground subject.

    Why: even after size-filtering, a photo can have background-crowd faces nearly
    as big as the main people (e.g. a graduation shot with a dense crowd behind).
    But rembg already separated foreground from background — so a face that belongs
    to a MAIN subject lands on the cutout silhouette, while a background face lands
    on the (removed) background. We sample the subject mask at each face's center
    and keep the face only if that point is part of the subject.

    `cut.mask` is the subject's alpha as a full-resolution "L" image (0-255). We
    map a normalized face center to a mask pixel and keep the face if alpha > 96.

    rembg keeps only its single largest blob, so in a photo with two foreground
    people who aren't touching it may include just one — which would drop the
    other person's face. To recover that case we ALSO keep any face that is at
    least 55% the size of the largest on-subject face: a genuine second foreground
    person has a similarly large face, while background-crowd faces are smaller
    (and were already thinned by detect_faces's size filter).

    If there is no cutout, we can't tell foreground from background, so we keep all
    faces unchanged. If nothing lands on the subject at all (e.g. rembg picked a
    different subject than the face detector), we keep the original faces rather
    than end up with nothing.
    """
    if cut is None or not faces:
        return faces
    mw, mh = cut.mask.width, cut.mask.height

    def area(f):
        return max(0.0, f[2] - f[0]) * max(0.0, f[3] - f[1])

    on_subject = []
    for f in faces:
        cx = min(mw - 1, int((f[0] + f[2]) / 2 * mw))
        cy = min(mh - 1, int((f[1] + f[3]) / 2 * mh))
        if cut.mask.getpixel((cx, cy)) > 96:
            on_subject.append(f)
    if not on_subject:
        return faces

    # Recover genuine foreground group members that rembg's single-blob mask missed
    # so the keep-box spans the WHOLE group (and the layout sizes a wide enough cell
    # to keep everyone): re-include any face nearly as big as the biggest kept face.
    # A real foreground person — even out at the edge of the row — has a similarly
    # large face, while background-crowd faces are smaller (and already thinned by
    # detect_faces's size filter), so they stay excluded. We keep this a size test
    # only (no "is it between the others" rule) precisely so a dense crowd standing
    # *behind* the main people isn't pulled back in.
    threshold = max(area(f) for f in on_subject) * 0.45
    kept = [f for f in faces if f in on_subject or area(f) >= threshold]
    return kept


def keep_box(faces, cut, margin=0.06):
    """Combine face boxes (and the subject box, if any) into a single "must-stay-in-
    frame" box, returned as (x0, y0, x1, y1) in 0-1, or None.

    Priority for what to protect when cropping/sizing a cell:
      1. If faces were found: first drop any that aren't on the foreground subject
         (_faces_on_subject), then take the bounding box enclosing the rest,
         expanded by a small `margin` (so we keep a little room around heads, not a
         tight crop at the hairline). This is the strongest signal.
      2. Else if there's a rembg cutout: fall back to the subject box (`cut.bbox`).
      3. Else: None -> the crop/layout treats it as "no preference" (center).

    The margin is added then clamped to [0,1] so it never exceeds the image.
    """
    faces = _faces_on_subject(faces, cut)
    if faces:
        x0 = min(f[0] for f in faces)
        y0 = min(f[1] for f in faces)
        x1 = max(f[2] for f in faces)
        y1 = max(f[3] for f in faces)
        return (max(0.0, x0 - margin), max(0.0, y0 - margin),
                min(1.0, x1 + margin), min(1.0, y1 + margin))
    if cut is not None:
        return cut.bbox
    return None


# ===========================================================================
# THE LAYOUT SKELETON
#
# Before we paste anything, we decide WHERE each photo goes. We do this by
# slicing the canvas rectangle into smaller rectangles ("cells"), one per photo,
# with no gaps and no overlaps — a "guillotine" layout. This is invisible
# scaffolding; build_collage later oversizes/jitters these cells so the photos
# actually overlap a little.
#
# The clever bit: we don't slice randomly. We search for the slicing that makes
# each cell's shape match its photo's subject as closely as possible, so photos
# barely need to be cropped and (especially) people don't get cut off. The next
# several small functions are the pieces of that search.
# ===========================================================================

def aspect_ratio(img):
    """Return width / height for a Pillow image. >1 = wide, <1 = tall, 1 = square.

    This single number is how we describe a photo's *shape* throughout the layout
    code. `img.size` is a (width, height) pair in pixels.
    """
    w, h = img.size
    return w / h


def hero_boost(k, boost):
    """How much extra area to give a branch of the layout that holds `k` photos.

    Why: a single photo sitting alone tends to look better a bit larger (a "hero"
    image) instead of cramped into a small equal share. `boost` (the --hero-boost
    value, default 1.5) is how much extra a lone photo gets:
      - k == 1 (one photo): full boost (e.g. 1.5x its "fair share").
      - k == 2 (a pair):     a smaller, partial boost.
      - k >= 3 (a group):    no boost (1.0); groups split their space evenly.
    The layout uses this as a *weight* when deciding how to divide space between
    two groups of photos.
    """
    if k == 1:
        return boost
    if k == 2:
        return 1.0 + (boost - 1.0) * 0.4
    return 1.0


def candidate_splits(n):
    """Which "split points" to try when dividing `n` photos into a left/right (or
    top/bottom) group.

    A split point `k` means "the first k photos go in one group, the rest in the
    other." We could try every k from 1..n-1, but for big groups that's a lot of
    combinations to search. So:
      - n <= 8: try them all (cheap, and gives the best layout).
      - n  > 8: try only a handful of sensible points (1, near halves, near
        thirds, n-1). The `{...}` is a set, which removes duplicates; `sorted`
        puts them back in order. This keeps the search fast on large albums while
        still exploring balanced and lopsided splits.
    """
    if n <= 8:
        return list(range(1, n))
    mid = n // 2
    return sorted({1, n - 1, mid, n // 3, 2 * n // 3})


def plan_layout(specs, weights, i, j, rect_aspect, boost, rng, memo):
    """Find the best way to slice a rectangle (shape `rect_aspect`) to hold the
    photos specs[i:j], and return (total_cost, plan).

    This is the heart of the layout. It is RECURSIVE: to lay out a group of
    photos in a rectangle, it tries cutting the rectangle in two (vertically or
    horizontally), then asks *itself* to lay out each half, and keeps whichever
    combination has the lowest total "cost." Cost = how badly photos' cell shapes
    mismatch their subjects (see leaf_cost). Lower cost = better fit = fewer/safer
    crops.

    Parameters:
      specs       : a list of per-photo "shape preferences" (from cell_spec). We
                    only look at the slice specs[i:j] (the photos in this branch).
      i, j        : start/end of the slice of photos this call is responsible for.
      rect_aspect : the shape (width/height) of the rectangle we're filling.
      boost       : the hero-boost value, passed down to hero_boost.
      rng         : the seeded random generator (used only to break near-ties).
      memo        : a cache (dictionary). See "memoization" note below.

    Returns a (cost, node) pair. A `node` describes the chosen plan as a little
    tree:
      ("leaf", i)                                  -> photo i fills this rectangle
      ("split", "v"|"h", k, fA, leftNode, rightNode)
          -> cut the rectangle ("v" = into left/right columns, "h" = into
             top/bottom rows) at fraction fA, with k photos on one side.
    realize_layout later walks this tree to produce actual pixel rectangles.

    Base case: if there's exactly one photo (n == 1), it fills the whole rectangle
    — that's a "leaf," and its cost is just how much its shape mismatches.

    Memoization (the `memo` cache): the same sub-problem (same photos in a rect of
    the same shape) can come up many times during the search. The cost only
    depends on the rectangle's *shape* (not its size), so we store answers keyed
    by (i, j, rounded aspect) and reuse them. Without this the search would be far
    slower. (We round the aspect so near-identical shapes share a cache entry.)
    """
    n = j - i
    if n == 1:
        # Only one photo here: it takes the whole rectangle. Cost = shape mismatch.
        return leaf_cost(rect_aspect, specs[i]), ("leaf", i)

    # Have we already solved this exact sub-problem? If so, reuse the answer.
    key = (i, j, round(rect_aspect, 3))
    if key in memo:
        return memo[key]

    # `eps` is a small tolerance: plans whose costs are within eps of each other
    # are treated as "basically tied," and we pick among them randomly (via rng).
    # That's what makes re-running with a different --seed give a different (but
    # still near-optimal) arrangement instead of the exact same one every time.
    eps = 0.05
    best_cost = None
    best_node = None
    for k in candidate_splits(n):
        # Split the photos into group A (first k) and group B (the rest).
        na, nb = k, n - k
        # Decide how much of the rectangle each group gets. Each group "wants"
        # area proportional to its photos' content weights (a flat screenshot
        # counts for less than a real photo), adjusted by hero_boost (a lone photo
        # gets extra). fa is group A's fraction of the rectangle.
        area_a = sum(weights[i:i + k]) * hero_boost(na, boost)
        area_b = sum(weights[i + k:j]) * hero_boost(nb, boost)
        fa = area_a / (area_a + area_b)

        # Now recurse for BOTH ways of cutting, and figure out each sub-piece's
        # shape:
        #  - vertical cut ("v"): two side-by-side columns. They share the full
        #    height, and split the width by `fa`, so a column's aspect = parent
        #    aspect * its width fraction.
        #  - horizontal cut ("h"): two stacked rows. They share the full width,
        #    and split the height by `fa`, so a row's aspect = parent aspect /
        #    its height fraction.
        cav, nav = plan_layout(specs, weights, i, i + k, rect_aspect * fa, boost, rng, memo)
        cbv, nbv = plan_layout(specs, weights, i + k, j, rect_aspect * (1 - fa), boost, rng, memo)
        cah, nah = plan_layout(specs, weights, i, i + k, rect_aspect / fa, boost, rng, memo)
        cbh, nbh = plan_layout(specs, weights, i + k, j, rect_aspect / (1 - fa), boost, rng, memo)

        # Compare this k's vertical option and horizontal option against the best
        # so far. Keep the cheapest; for near-ties, flip a coin (rng) so we get
        # variety across seeds.
        for cost, node in (
            (cav + cbv, ("split", "v", k, fa, nav, nbv)),
            (cah + cbh, ("split", "h", k, fa, nah, nbh)),
        ):
            if best_cost is None or cost < best_cost - eps:
                best_cost, best_node = cost, node
            elif cost < best_cost + eps and rng.random() < 0.5:
                best_cost, best_node = min(best_cost, cost), node

    memo[key] = (best_cost, best_node)
    return memo[key]


def realize_layout(node, images_sorted, x0, y0, x1, y1):
    """Turn the abstract plan `node` (from plan_layout) into actual pixel
    rectangles, one per photo.

    plan_layout decided the *structure* (where to cut, what fraction). This walks
    that structure with concrete coordinates and produces the final list:
        [(path, img, x0, y0, x1, y1), ...]
    where (x0, y0) is the top-left corner and (x1, y1) the bottom-right corner of
    each photo's cell, in canvas pixels.

    It is recursive in the same shape as the plan:
      - a "leaf" returns the one photo filling the current rectangle.
      - a "split" computes the cut line and recurses into the two sub-rectangles.
        For "v" (columns) the cut is a vertical line at split_x; for "h" (rows)
        it's a horizontal line at split_y. The two halves share the exact cut
        coordinate, so there are no gaps or overlaps between cells.

    `round(...)` converts the fractional cut position to a whole pixel. Because
    both sides use the SAME rounded split coordinate, the cells still line up
    perfectly (no 1-pixel seam).
    """
    if node[0] == "leaf":
        path, img = images_sorted[node[1]]
        return [(path, img, x0, y0, x1, y1)]
    _, orient, _k, fa, left, right = node
    if orient == "v":
        split_x = x0 + round((x1 - x0) * fa)
        return (realize_layout(left, images_sorted, x0, y0, split_x, y1)
                + realize_layout(right, images_sorted, split_x, y0, x1, y1))
    split_y = y0 + round((y1 - y0) * fa)
    return (realize_layout(left, images_sorted, x0, y0, x1, split_y)
            + realize_layout(right, images_sorted, x0, split_y, x1, y1))


def content_weight(img):
    """Area weight in (0, 1] for a photo — lower for flat, low-information tiles
    like UI screenshots, so real photos get more of the canvas.

    Why: a screen capture (a chat UI, a FAQ page, a terminal) is mostly large
    uniform regions; giving it a full-size cell wastes space a real photo would use
    better. We measure "flatness" as the fraction of 8x8 blocks (on a 64x64
    thumbnail) that are near-uniform: a real photo has texture nearly everywhere
    (low fraction), a screenshot has many flat blocks (high fraction). Calibrated on
    the test set — every real photo scored <=0.25, the one true UI screenshot 0.53 —
    so we only start trimming above 0.30 and never below 0.6 weight. That leaves all
    normal photos at weight 1.0 (layout unchanged), trimming only clear screenshots.
    """
    g = np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32)
    flat = np.mean([g[r:r + 8, c:c + 8].std() < 6.0
                    for r in range(0, 64, 8) for c in range(0, 64, 8)])
    t = min(1.0, max(0.0, (flat - 0.30) / 0.25))   # 0 at flat<=0.30, 1 at flat>=0.55
    return 1.0 - 0.4 * t                            # 1.0 (photo) .. 0.6 (screenshot)


def cell_spec(img, box):
    """Describe a photo's *shape preference* for the layout, as
    (target_aspect, narrow_penalty, wide_penalty).

    This is where we tell the layout "please don't cut this person off." The
    layout will fit each photo's cell shape to its `target_aspect`, but no canvas
    can match every photo perfectly, so some cropping is unavoidable. The two
    penalties say which *direction* of mismatch is worse for THIS photo:

      - A cell NARROWER than the photo crops the photo's LEFT/RIGHT edges. In a
        group photo, the outer people's faces live on the left and right — so
        side-cropping is bad. narrow_penalty grows with the keep-region's WIDTH.
      - A cell WIDER than the photo crops the TOP/BOTTOM. That's usually sky,
        floor, or torsos — less bad. wide_penalty grows with its HEIGHT.

    `box` is the "must-stay-in-frame" region in 0-1 coords (from keep_box: the
    faces if any, else the rembg subject). A wide two-/three-face photo therefore
    has a wide box -> a large narrow_penalty -> the layout strongly prefers a WIDE
    cell so nobody gets chopped. The narrow penalty is multiplied by 2.8 to lean
    even harder against slicing the sides (where outer faces sit).

    Edge cases:
      - box is None (no faces and no cutout, e.g. --no-cutout on an object photo):
        we have no region to protect, so we fall back to symmetric penalties
        (1.0, 1.0) and just match the image's own aspect.
      - max(0.4, ...) keeps a minimum penalty so even a tiny region still nudges
        the layout (and avoids a zero penalty that the cost search would ignore).
    """
    a = aspect_ratio(img)
    if box is None:
        return (a, 1.0, 1.0)
    bw = box[2] - box[0]   # keep-region width fraction  (right - left)
    bh = box[3] - box[1]   # keep-region height fraction (bottom - top)
    return (a, max(0.4, bw * 1.6), max(0.4, bh))


def leaf_cost(rect_aspect, spec):
    """How "bad" it is to put a photo (described by `spec`) into a cell of shape
    `rect_aspect`. Bigger number = worse fit. plan_layout adds these up and
    minimizes the total.

    `spec` is (target_aspect a, narrow_penalty, wide_penalty) from cell_spec.

    We compare shapes using logarithms so that, say, "twice as wide" and "twice
    as tall" count as equal-sized mismatches (log makes ratios symmetric). `d` is
    the signed mismatch:
      - d < 0: the cell is NARROWER than the photo wants -> it will crop the
        sides -> charge the narrow_penalty.
      - d > 0: the cell is WIDER -> it will crop top/bottom -> charge wide_penalty.
    This asymmetry is the whole point: the same amount of mismatch costs more in
    the face-cropping direction, so the layout avoids it.
    """
    a, narrow_w, wide_w = spec
    d = math.log(rect_aspect) - math.log(a)
    return narrow_w * -d if d < 0 else wide_w * d


def layout_cells(images, width, height, rng, boost, keep_boxes=None):
    """Produce the final skeleton: a list of cells [(path, img, x0,y0,x1,y1), ...]
    that tile the whole canvas, each shaped to fit its photo's keep-region.

    Steps:
      - If there's only ONE photo, it simply fills the entire canvas (no slicing
        needed). This is an important edge case — the search below assumes 2+.
      - Build a `spec` (shape preference) for each photo via cell_spec, looking up
        its keep-box in `keep_boxes` (a dict keyed by the file path as text). A
        keep-box is the face region if faces were found, else the rembg subject,
        else None. `keep_boxes` may be None; then specs fall back to image aspect.
      - Sort the photos by their target aspect. Sorting groups similar shapes
        next to each other, which the guillotine search can then keep clustered
        (e.g. all the tall photos in one column). plan_layout splits *contiguous*
        ranges, so the sort order matters.
      - "Best-of-N": plan_layout already finds a near-best plan, but its random
        tie-breaking means different runs find different near-best plans. So we
        run it 8 times (each consuming the rng a little differently), keep every
        plan within 4% of the lowest cost, and pick one of those at random. This
        keeps the *quality* high (only tight fits qualify) while letting a different
        --seed actually pick a different arrangement (real variety, not the same
        strict-minimum plan every time). The `{}` passed as `memo` is a fresh empty
        cache for each attempt.
      - Finally, realize_layout turns the winning plan into real pixel rectangles.
    """
    n = len(images)
    if n == 1:
        path, img = images[0]
        return [(path, img, 0, 0, width, height)]

    def spec_of(item):
        path, img = item
        return cell_spec(img, keep_boxes.get(str(path)) if keep_boxes else None)

    items = sorted(images, key=lambda it: spec_of(it)[0])  # cluster by image aspect
    specs = [spec_of(it) for it in items]
    weights = [content_weight(img) for _, img in items]    # flat tiles get less area

    candidates = []
    for _ in range(8):
        cost, node = plan_layout(specs, weights, 0, n, width / height, boost, rng, {})
        candidates.append((cost, node))
    # Keep every plan within 4% of the best cost, then pick one at random (via rng).
    # This keeps quality high (only near-optimal fits qualify) while letting a
    # different --seed actually land on a different arrangement, instead of always
    # collapsing to the single strict-minimum plan (which made seeds look identical).
    best_cost = min(cost for cost, _ in candidates)
    near_best = [node for cost, node in candidates if cost <= best_cost * 1.04]
    best_node = rng.choice(near_best)
    return realize_layout(best_node, items, 0, 0, width, height)


def cover_crop(img, target_w, target_h):
    """Resize-and-crop `img` so it EXACTLY fills a target_w x target_h box, keeping
    the center (this is the same idea as CSS `object-fit: cover`).

    Why: a cell is a fixed-size box, but the photo is a different shape. To fill
    the box without distorting (stretching) the photo, we scale the photo up until
    it covers the box on both sides, then trim the overflow off the edges.

    How:
      - `scale = max(...)` picks the *larger* of the two scale factors so the
        photo covers the box completely (the other dimension will overflow).
      - resize to that scale with LANCZOS (high quality).
      - crop a centered target_w x target_h rectangle out of the scaled image.

    Edge cases: `max(1, ...)` everywhere prevents zero or negative sizes (which
    Pillow would reject) if a cell ends up tiny. Used for the dimmed backdrop and
    as the building block other crops rely on.
    """
    target_w, target_h = max(1, target_w), max(1, target_h)
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


# A `Cutout` bundles everything we learn about one photo's subject. A namedtuple
# is just a lightweight record: you access fields by name (cut.bbox, cut.mask,
# ...) instead of remembering positions. Fields:
#   rgba      — the cutout image: subject in full color, background transparent.
#   mask      — a black/white ("L") image of just the subject's shape (its alpha).
#   bbox      — the subject's bounding box (x0, y0, x1, y1), as fractions 0-1 of
#               the image, so it's independent of the photo's pixel size.
#   centroid  — the center point of that box, also as fractions 0-1.
#   isolation — how "clean" the cutout is: the fraction of the selected pixels
#               that form ONE connected blob (1.0 = a single clean subject; lower
#               means the AI also grabbed stray bits).
#   coverage  — what fraction of the whole photo the subject fills.
# The last few (bbox, centroid, isolation, coverage) are computed once here and
# reused later for framing the photo and for choosing/limiting the "pop."
Cutout = namedtuple("Cutout", "rgba mask bbox centroid isolation coverage")


def subject_cutout(img, session):
    """Run the AI background remover on one photo and return a cleaned-up `Cutout`,
    or None if it didn't find a usable subject.

    `session` is a reusable rembg model handle (created once in main and passed in
    so we don't reload the model for every photo).

    Walkthrough:
      1. `remove(...)` asks rembg to cut out the foreground. `post_process_mask`
         tells rembg to tidy the edges. The result is an RGBA image (subject
         solid, background transparent). If rembg throws (bad input, model issue),
         we warn and return None instead of crashing.
      2. We pull out the Alpha channel (transparency) and turn it into a True/False
         map `binary`: True where the subject is reasonably solid (alpha > 96).
      3. Degenerate-mask guard: if the subject is almost nothing (<2% of the
         frame) or almost everything (>97%), it isn't a meaningful "subject" to
         feature — return None and treat the photo as having no cutout.
      4. "Largest connected component": the AI sometimes leaves stray specks far
         from the main subject. Using scipy we label every separate blob, find the
         biggest one, and keep only that — dropping the islands. `isolation`
         records how much of the original selection was that main blob (a quality
         signal used later). If scipy isn't available, we skip this step
         gracefully (the try/except) rather than fail.
      5. Bounding box: `np.where` finds all True pixels; their min/max x and y
         give the box around the subject. (If somehow nothing is left, return
         None.) We store the box as fractions of the image size so it's
         resolution-independent.
      6. Clean mask + soft edge: rebuild the alpha keeping only the main blob,
         apply a MedianFilter (removes lone speckles) and a tiny GaussianBlur
         (softens the cut edge by ~1.5px so the sticker doesn't look harshly
         scissored), and attach it back as the cutout's transparency.
      7. Return the Cutout record with all the measurements.
    """
    from rembg import remove
    try:
        rgba = remove(img, session=session, post_process_mask=True).convert("RGBA")
    except Exception as e:
        print(f"  ! cutout failed: {e}", file=sys.stderr)
        return None

    # alpha: a 2D grid of 0-255 transparency values. binary: True where solid.
    alpha = np.asarray(rgba.getchannel("A"))
    binary = alpha > 96
    total = binary.size  # total pixel count = width * height
    if not (0.02 <= binary.sum() / total <= 0.97):
        return None  # nothing meaningful, or basically the whole frame

    # Drop stray islands: keep only the single largest connected region.
    isolation = 1.0
    try:
        from scipy import ndimage
        labels, n = ndimage.label(binary)       # number every separate blob
        if n > 1:
            counts = np.bincount(labels.ravel())  # how many pixels per blob
            counts[0] = 0                          # blob 0 is the background; ignore
            biggest = int(counts.argmax())         # id of the largest blob
            keep = labels == biggest               # True only on that blob
            isolation = keep.sum() / max(1, int(binary.sum()))
            binary = keep
    except Exception:
        pass  # scipy missing or errored: skip island-removal, not fatal

    # Bounding box of the remaining subject pixels.
    ys, xs = np.where(binary)   # row (y) and column (x) indices of True pixels
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    iw, ih = img.width, img.height

    # Build the cleaned, soft-edged alpha mask and apply it to the cutout.
    clean = np.where(binary, alpha, 0).astype("uint8")  # keep alpha on the blob, 0 elsewhere
    mask = Image.fromarray(clean, "L").filter(ImageFilter.MedianFilter(3))
    mask = mask.filter(ImageFilter.GaussianBlur(1.5))   # small, clean feather
    rgba.putalpha(mask)

    return Cutout(
        rgba=rgba,
        mask=mask,
        bbox=(x0 / iw, y0 / ih, x1 / iw, y1 / ih),
        centroid=((x0 + x1) / 2 / iw, (y0 + y1) / 2 / ih),
        isolation=isolation,
        coverage=binary.sum() / total,
    )


def cover_transform(img, target_w, target_h, bbox=None, pan=(0, 0)):
    """Work out HOW to scale and crop `img` to fill a target_w x target_h box —
    but return the numbers instead of doing it, so two different pastes can share
    the exact same transform.

    Returns (nw, nh, left, top):
      nw, nh   = the size to scale the FULL image to.
      left, top = the top-left corner of the crop window within that scaled image.

    Why return numbers? The photo's BACKGROUND (subject_cover_crop) and its
    SUBJECT cutout must line up pixel-for-pixel when we later lift the subject on
    top. By computing the transform once here and reusing it for both, the cutout
    sits exactly where the subject is in the background — no ghosty double image.

    Two modes:
      - bbox is None: a plain centered cover crop (trim equal amounts off each
        side). Same math as cover_crop.
      - bbox given (the subject's box): we slide the crop window so the subject
        stays in frame instead of being trimmed off. We aim the window at the
        box's center, then clamp it so the box's edges stay inside the window *if
        the box fits*; the final `max(0, min(...))` clamps the window to stay
        within the scaled image so we never crop outside the photo.
    """
    target_w, target_h = max(1, target_w), max(1, target_h)
    s = max(target_w / img.width, target_h / img.height)        # cover scale factor
    nw, nh = max(1, round(img.width * s)), max(1, round(img.height * s))
    px, py = pan   # user pan (GUI), added to the final clamped window, then clamped again
    if bbox is None:
        left, top = (nw - target_w) // 2 + px, (nh - target_h) // 2 + py
        return nw, nh, max(0, min(nw - target_w, left)), max(0, min(nh - target_h, top))
    # Subject box in scaled-image pixels:
    bx0, by0, bx1, by1 = bbox[0] * nw, bbox[1] * nh, bbox[2] * nw, bbox[3] * nh
    # Horizontal: center the window on the box, then nudge to keep the box inside.
    left = round((bx0 + bx1) / 2 - target_w / 2)
    left = min(max(left, round(bx1) - target_w), round(bx0))   # keep bbox inside if it fits
    left = max(0, min(nw - target_w, left))                    # stay within the image
    left = max(0, min(nw - target_w, left + px))               # user pan, still within the image
    # Vertical: same idea.
    top = round((by0 + by1) / 2 - target_h / 2)
    top = min(max(top, round(by1) - target_h), round(by0))
    top = max(0, min(nh - target_h, top))
    top = max(0, min(nh - target_h, top + py))
    return nw, nh, left, top


def subject_cover_crop(img, target_w, target_h, bbox=None):
    """Actually produce the cropped photo that fills a cell, keeping the subject in
    frame. Thin wrapper: get the transform from cover_transform, then resize the
    image and cut out the chosen window. With bbox=None it's a plain centered crop.
    """
    nw, nh, left, top = cover_transform(img, target_w, target_h, bbox)
    scaled = img.resize((nw, nh), Image.LANCZOS)
    return scaled.crop((left, top, left + max(1, target_w), top + max(1, target_h)))


# ---------------------------------------------------------------------------
# Helpers that build "maps" — boolean numpy grids the size of the whole canvas,
# where each pixel is True/False. We use these to reason about WHERE things are:
# which pixels a photo's rectangle covers, and which pixels its subject occupies.
# Combining maps with `&` (overlap), `|` (union), `~` (opposite) lets us answer
# questions like "would this subject get covered?" with simple, fast math.
# ---------------------------------------------------------------------------

def _rect_mask(px0, py0, pw, ph, width, height):
    """Return a canvas-sized True/False map that is True inside one photo's
    rectangle (top-left px0,py0, size pw x ph) and False elsewhere.

    The `max(0, ...)` clamps the indices so a rectangle that pokes off the canvas
    edge (negative position, or extending past the right/bottom) just fills the
    visible part instead of erroring. (Slicing a numpy array past its end is safe;
    a negative start is what we guard against.)
    """
    rect = np.zeros((height, width), dtype=bool)
    rect[max(0, py0):max(0, py0 + ph), max(0, px0):max(0, px0 + pw)] = True
    return rect


def _footprint(mask, nw, nh, posx, posy, width, height):
    """Return a canvas-sized True/False map of where a subject `mask` lands when
    placed at the given transform.

    We make a blank black canvas-sized "L" image, paste the (resized) subject mask
    onto it at (posx, posy) — Pillow safely clips anything off-edge — then convert
    to a boolean map (True where the pasted mask is reasonably solid, >96). This
    tells us exactly which canvas pixels that subject would occupy.
    """
    layer = Image.new("L", (width, height), 0)
    layer.paste(mask.resize((nw, nh), Image.LANCZOS), (posx, posy))
    return np.asarray(layer) > 96


class Plan:
    """A collage once its layout is decided: everything fixed until the seed or a
    layout setting changes. Rendering a Plan is cheap and can be repeated with
    per-photo offsets (the GUI's drag/pan), without redoing any layout or AI work.

    placements : list of (path, img, px0, py0, pw, ph), the overlapping frames
    order      : placement indices, paint order back-to-front
    selected   : placement indices chosen to pop (frozen at plan time)
    cache      : resized images/masks reused across renders
    """

    def __init__(self, width, height, backdrop, placements, order, cutouts, keep_boxes):
        self.width, self.height, self.backdrop = width, height, backdrop
        self.placements, self.order = placements, order
        self.cutouts, self.keep_boxes = cutouts, keep_boxes
        self.selected = []
        self.fp0 = None    # footprints at zero offsets / full scale (reused by render)
        self.cache = {}

    def cut_of(self, idx):
        return self.cutouts.get(str(self.placements[idx][0])) if self.cutouts else None

    def keep_of(self, idx):
        return self.keep_boxes.get(str(self.placements[idx][0])) if self.keep_boxes else None


def _resized(plan, kind, idx, im, size):
    """Cached `im.resize(size, LANCZOS)`; the resize dominates render time and
    doesn't change when a photo is only moved."""
    key = (kind, idx, size)
    hit = plan.cache.get(key)
    if hit is None:
        if len(plan.cache) > 96:
            plan.cache.clear()
        hit = plan.cache[key] = im.resize(size, Image.LANCZOS)
    return hit


def _geometry(plan, offsets, scale):
    """Per-placement (x, y, w, h, pan) in canvas pixels at `scale`, with the
    user's `offsets` ({idx: (dx, dy, panx, pany)}, full-res units) applied."""
    geo = []
    for idx, (_, _, px0, py0, pw, ph) in enumerate(plan.placements):
        dx, dy, panx, pany = offsets.get(idx, (0, 0, 0, 0))
        geo.append((round((px0 + dx) * scale), round((py0 + dy) * scale),
                    max(1, round(pw * scale)), max(1, round(ph * scale)),
                    (round(panx * scale), round(pany * scale))))
    return geo


def _footprints(plan, geo, W, H):
    """Section C: for each photo, the maps used to protect and pop subjects.
      own[idx]     : the photo's rectangle
      face_ft[idx] : where the subject actually SHOWS (silhouette clipped to the rect)
      pop_ft[idx]  : the LIFTED silhouette (can spill past the rect)
      lift_place[idx] : (w, h, x, y) of the lifted cutout, scaled up about its centroid
    """
    own, face_ft, pop_ft, lift_place = {}, {}, {}, {}
    for idx, (path, img, _, _, _, _) in enumerate(plan.placements):
        x, y, w, h, pan = geo[idx]
        own[idx] = _rect_mask(x, y, w, h, W, H)
        cut = plan.cut_of(idx)
        if cut is None:
            continue
        # Same framing transform the background uses, so the footprint lines up
        # with where the subject actually lands in the painted photo.
        fnw, fnh, fl, ftop = cover_transform(img, w, h, plan.keep_of(idx), pan)
        bx, by = x - fl, y - ftop          # top-left of the (unscaled) cutout
        mask = _resized(plan, "m", idx, cut.mask, (fnw, fnh))
        full = _footprint(mask, fnw, fnh, bx, by, W, H)
        scx = bx + cut.centroid[0] * fnw   # subject centroid on canvas
        scy = by + cut.centroid[1] * fnh
        lnw, lnh = round(fnw * POP_LIFT), round(fnh * POP_LIFT)
        lx = round(scx - cut.centroid[0] * lnw)
        ly = round(scy - cut.centroid[1] * lnh)
        lift_place[idx] = (lnw, lnh, lx, ly)
        lmask = _resized(plan, "m", idx, cut.mask, (lnw, lnh))
        pop_ft[idx] = _footprint(lmask, lnw, lnh, lx, ly, W, H)
        face_ft[idx] = full & own[idx]
    return own, face_ft, pop_ft, lift_place


def _select_pops(plan, own, face_ft, pop_ft, has_face, pop, pop_threshold):
    """Section D: quality-gated pop selection (see the CLAUDE.md pipeline notes).
    Returns the chosen placement indices."""
    height, width = plan.height, plan.width
    selected = []
    if plan.cutouts and pop > 0:
        scored = []
        for idx in face_ft:
            fa = face_ft[idx]
            fsum = int(fa.sum())      # how many pixels this subject covers
            if fsum == 0:
                continue              # no visible subject; can't pop it

            # Pops feature real PEOPLE only: a subject needs a detected on-subject
            # face to be a candidate (posters/screens/objects always read as gimmicky).
            if not (has_face and has_face.get(str(plan.placements[idx][0]))):
                continue

            # "seam": fraction of this subject inside a neighbor's rectangle.
            neighbors = np.zeros((height, width), dtype=bool)
            for j in own:
                if j != idx:
                    neighbors |= own[j]
            seam = (fa & neighbors).sum() / fsum

            # "collision": worst fraction of ANY single OTHER subject this pop
            # would cover (using the pop's full lifted reach).
            collision = 0.0
            for j in face_ft:
                if j != idx and face_ft[j].any():
                    covered = int((pop_ft[idx] & face_ft[j]).sum()) / int(face_ft[j].sum())
                    collision = max(collision, covered)

            # Score: clean cutout (isolation) x well-sized (peaks at ~35% of its
            # photo) x leans over a neighbor (seam) x not a full-frame screenshot.
            cut = plan.cut_of(idx)
            size_fit = max(0.0, 1.0 - abs(cut.coverage - 0.35) / 0.35)
            bw = max(1e-6, cut.bbox[2] - cut.bbox[0])
            bh = max(1e-6, cut.bbox[3] - cut.bbox[1])
            fill_ratio = cut.coverage / (bw * bh)
            rect_factor = max(0.0, 1.0 - max(0.0, fill_ratio - 0.72) / 0.28)
            score = cut.isolation * size_fit * min(1.0, seam * 4.0) * rect_factor
            scored.append((score, collision, idx))

        # Highest score first; a pop must clear the threshold, cover <=6% of any
        # other subject, and not overlap an already-chosen pop by >15%. `pop` is a
        # hard ceiling.
        scored.sort(reverse=True)
        for score, collision, idx in scored:
            if len(selected) >= pop or score < pop_threshold:
                break
            if collision > 0.06:
                continue
            clutter = False
            for j in selected:
                inter = int((pop_ft[idx] & pop_ft[j]).sum())
                smaller = min(int(pop_ft[idx].sum()), int(pop_ft[j].sum()))
                if smaller > 0 and inter / smaller > 0.15:
                    clutter = True
                    break
            if not clutter:
                selected.append(idx)
    return selected


def plan_collage(images, width, ratio, boost, rng, overlap, scatter, cutouts, pop,
                 keep_boxes=None, has_face=None, pop_threshold=0.25):
    """Decide the layout and return a Plan (no painting yet).

    Inputs: the loaded `images`, canvas `width` and shape `ratio`, hero `boost`,
    the seeded `rng`, the `overlap`/`scatter` look knobs, the `cutouts` dict
    (path -> Cutout or None; may be None), `pop` (max subjects lifted on top),
    `keep_boxes` (path -> must-stay-in-frame box), `has_face` (path -> bool) and
    `pop_threshold` (min tastefulness score to pop).

    A. canvas size + dimmed blurred backdrop (so jitter gaps read as depth)
    B. skeleton layout, then oversize + jitter each cell into a placement
    C/D. footprints, then pick which subjects pop
    """
    # --- A. Canvas + backdrop ----------------------------------------------
    height = round(width / ratio)
    cells = layout_cells(images, width, height, rng, boost, keep_boxes)

    backdrop = cover_crop(images[0][1], width, height)
    backdrop = backdrop.filter(ImageFilter.GaussianBlur(max(width, height) // 12))
    backdrop = backdrop.point(lambda p: int(p * 0.55))

    # --- B. Turn each cell into an overlapping, slightly-jittered placement --
    placements = []
    for path, img, x0, y0, x1, y1 in cells:
        cw, ch = x1 - x0, y1 - y0                 # the skeleton cell's size
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2     # its center
        pw, ph = max(1, round(cw * overlap)), max(1, round(ch * overlap))
        # Jitter the center by a seeded random amount, at most half the extra
        # size times `scatter`. (rng.random()*2 - 1) is in [-1, 1].
        jx = (rng.random() * 2 - 1) * (pw - cw) * 0.5 * scatter
        jy = (rng.random() * 2 - 1) * (ph - ch) * 0.5 * scatter
        px0 = round(cx + jx - pw / 2)
        py0 = round(cy + jy - ph / 2)
        placements.append((path, img, px0, py0, pw, ph))

    # Paint back-to-front: biggest cells first so small photos aren't buried.
    order = sorted(range(len(placements)),
                   key=lambda i: placements[i][4] * placements[i][5], reverse=True)
    plan = Plan(width, height, backdrop, placements, order, cutouts, keep_boxes)

    # --- C/D. Footprints at zero offset, then choose the pops ---------------
    plan.fp0 = _footprints(plan, _geometry(plan, {}, 1.0), width, height)
    own, face_ft, pop_ft, _ = plan.fp0
    plan.selected = _select_pops(plan, own, face_ft, pop_ft, has_face, pop, pop_threshold)

    if cutouts:
        names = [Path(placements[i][0]).name for i in plan.selected]
        print(f"  popped {len(names)}: {', '.join(names) if names else 'none'}")
    return plan


def pan_limits(plan, idx):
    """((min_x, max_x), (min_y, max_y)) pan range for placement `idx` in full-res
    pixels. An axis with no slack (photo and frame share that aspect) is (0, 0)."""
    _, img, _, _, pw, ph = plan.placements[idx]
    nw, nh, left, top = cover_transform(img, pw, ph, plan.keep_of(idx))
    return (-left, nw - pw - left), (-top, nh - ph - top)


def render_plan(plan, scale=1.0, offsets=None):
    """Paint a Plan and return the image. `offsets` is {idx: (dx, dy, panx, pany)}
    in full-res pixels (move the frame / pan the picture inside it); `scale` < 1
    renders a cheap preview. Footprints (and so protected compositing and pop
    lift) are recomputed from the moved frames, so no overlap covers a subject.
    """
    offsets = offsets or {}
    W, H = round(plan.width * scale), round(plan.height * scale)
    geo = _geometry(plan, offsets, scale)
    if not offsets and scale == 1.0 and plan.fp0 is not None:
        own, face_ft, pop_ft, lift_place = plan.fp0
    else:
        own, face_ft, pop_ft, lift_place = _footprints(plan, geo, W, H)
    canvas = plan.backdrop.copy() if scale == 1.0 else plan.backdrop.resize((W, H), Image.BILINEAR)

    # --- E. Pass 1: paint backgrounds, never covering an existing subject ---
    # `protected` = every subject painted so far; the next photo's paste mask is
    # 0 there, so it paints around placed subjects (their own photo shows through).
    protected = np.zeros((H, W), dtype=bool)
    for idx in plan.order:
        path, img, _, _, _, _ = plan.placements[idx]
        x, y, w, h, pan = geo[idx]
        nw, nh, left, top = cover_transform(img, w, h, plan.keep_of(idx), pan)
        scaled = _resized(plan, "i", idx, img, (nw, nh))
        bg = scaled.crop((left, top, left + w, top + h))
        mask = np.full((h, w), 255, dtype=np.uint8)
        cx0, cy0 = max(0, x), max(0, y)
        cx1, cy1 = min(W, x + w), min(H, y + h)
        if cx1 > cx0 and cy1 > cy0:
            mask[cy0 - y:cy1 - y, cx0 - x:cx1 - x][protected[cy0:cy1, cx0:cx1]] = 0
        canvas.paste(bg, (x, y), Image.fromarray(mask, "L"))
        if idx in face_ft:
            protected |= face_ft[idx]

    # --- F. Pass 2: lift the featured subjects on top with a soft shadow ----
    for idx in plan.order:
        if idx not in plan.selected or idx not in lift_place:
            continue
        cut = plan.cut_of(idx)
        lnw, lnh, x, y = lift_place[idx]
        sub = _resized(plan, "r", idx, cut.rgba, (lnw, lnh))
        # Shadow: blurred, faint black copy of the alpha, offset down-right.
        blur = max(5, max(W, H) // 90)
        sh_dx, sh_dy = round(9 * scale), round(11 * scale)
        shadow = Image.new("RGBA", sub.size, (0, 0, 0, 0))
        shadow.putalpha(sub.getchannel("A").filter(ImageFilter.GaussianBlur(blur))
                        .point(lambda p: int(p * 0.4)))
        canvas.paste(shadow, (x + sh_dx, y + sh_dy), shadow)
        canvas.paste(sub, (x, y), sub)

    return canvas


def build_collage(images, width, ratio, boost, rng, overlap, scatter, cutouts, pop,
                  keep_boxes=None, has_face=None, pop_threshold=0.25):
    """Plan and render a collage in one go (the CLI path). See plan_collage."""
    return render_plan(plan_collage(images, width, ratio, boost, rng, overlap, scatter,
                                    cutouts, pop, keep_boxes, has_face, pop_threshold))


Analysis = namedtuple("Analysis", "images cutouts keep_boxes has_face")

_REMBG_SESSIONS = {}   # model name -> rembg session (the model load is slow)


def _rembg_session(model):
    if model not in _REMBG_SESSIONS:
        from rembg import new_session
        _REMBG_SESSIONS[model] = new_session(model)
    return _REMBG_SESSIONS[model]


def analyze(images, opts, on_progress=None):
    """The slow, AI-heavy half of the pipeline: everything that depends only on the
    photos and the analysis settings, not on layout. Returns an Analysis.

    `images` is the loaded list of (path, img). `opts` needs the attributes
    no_color, color_strength, no_cutout, model and no_faces (argparse Namespace or
    anything similar). `on_progress(message, fraction)` is called as it goes.
    Every step degrades gracefully: a failure warns and continues.
    """
    def say(msg, frac):
        if on_progress:
            on_progress(msg, frac)

    # Color cohesion first, so cutouts and the backdrop all inherit the grade.
    if not opts.no_color:
        say("Matching colors", 0.02)
        try:
            images = unify_colors(images, opts.color_strength)
        except Exception as e:
            print(f"  ! color cohesion unavailable ({e}); using original colors",
                  file=sys.stderr)

    # Cutouts: one shared session avoids reloading the ~200MB model per image.
    cutouts = None
    if not opts.no_cutout:
        try:
            say("Loading cutout model", 0.05)
            session = _rembg_session(opts.model)
            cutouts = {}
            for i, (path, img) in enumerate(images):
                say(f"Cutting out {i + 1}/{len(images)}", 0.1 + 0.6 * i / len(images))
                cut = subject_cutout(img, session)
                cutouts[str(path)] = cut
                print(f"  cutout {path.name}: {'ok' if cut is not None else 'skipped'}")
        except Exception as e:
            print(f"  ! cutouts unavailable ({e}); using plain overlap only",
                  file=sys.stderr)
            cutouts = None

    # keep_boxes: the region that must stay in frame (faces if found, else the
    # subject). Computed for ALL photos so face-aware framing works with
    # --no-cutout. has_face lets pop selection favor people over objects.
    keep_boxes, has_face = {}, {}
    for i, (path, img) in enumerate(images):
        say(f"Finding faces {i + 1}/{len(images)}", 0.7 + 0.3 * i / len(images))
        faces = [] if opts.no_faces else detect_faces(img)
        cut = cutouts.get(str(path)) if cutouts else None
        on_subject = _faces_on_subject(faces, cut) if faces else []
        keep_boxes[str(path)] = keep_box(faces, cut)
        has_face[str(path)] = bool(on_subject)
        if faces:
            print(f"  faces {path.name}: {len(on_subject)}/{len(faces)} on subject")
    say("Analysis done", 1.0)
    return Analysis(images, cutouts, keep_boxes, has_face)


def resolve_output_path(path: Path) -> Path:
    """Pick a filename that doesn't overwrite an existing file.

    If "output/collage.jpg" is free, use it. If it already exists, try
    "output/collage2.jpg", then "collage3.jpg", and so on until one is free. This
    way repeated runs accumulate (collage.jpg, collage2.jpg, ...) instead of
    clobbering your previous result. `stem` is the name without extension,
    `suffix` is ".jpg", `parent` is the folder.
    """
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    n = 2
    while True:
        candidate = parent / f"{stem}{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def main():
    """The entry point: tie everything together and write the file.

    Order of operations (with the guard checks that make it robust):
      1. Parse options.
      2. Check the input folder exists; if not, print an error and exit.
      3. Find image files; if none, error and exit.
      4. Create the seeded random generator (random.Random(seed)); if --seed was
         not given it's None, which means "truly random each run."
      5. Load the images; if none could be opened, error and exit.
      6. Parse the canvas ratio; on a bad value, print the message and exit.
      7. Unless --no-color: grade every photo toward one shared look (unify_colors)
         so the set feels cohesive. This happens BEFORE cutouts/faces so the graded
         pixels flow into everything downstream.
      8. Unless --no-cutout: load the rembg model ONCE (new_session) and run the
         cutout on every photo, storing results in `cutouts` keyed by file path.
         If rembg isn't installed or the model fails to load, we warn and fall
         back to cutouts = None (the collage still works, just with no AI cutouts).
      9. Build `keep_boxes` (path -> must-stay-in-frame region): detect faces in
         each photo (unless --no-faces) and combine with the subject box. This is
         what framing and cell-sizing use to keep faces in view.
     10. Build the collage.
     11. Pick a non-clobbering output path, make sure its folder exists
         (mkdir parents=True), and save. We only pass JPEG `quality` when the
         output is actually a .jpg/.jpeg.
    Each `sys.exit(1)` ends the program with a non-zero status (the convention for
    "something went wrong").
    """
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    paths = find_images(input_dir, args.recursive)
    if not paths:
        print(f"No supported images found in '{input_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(paths)} image(s) in '{input_dir}'.")

    if len(paths) > args.max_photos:
        print(f"{len(paths)} images exceeds the cap of {args.max_photos}; a collage "
              f"this dense gets cramped. Pass fewer images or raise --max-photos.",
              file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    images = load_images(paths, args.order, rng)
    if not images:
        print("No images could be loaded.", file=sys.stderr)
        sys.exit(1)

    try:
        ratio = parse_ratio(args.ratio)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    analysis = analyze(images, args)
    images = analysis.images

    collage = build_collage(
        images=images,
        width=args.width,
        ratio=ratio,
        boost=args.hero_boost,
        rng=rng,
        overlap=args.overlap,
        scatter=args.scatter,
        cutouts=analysis.cutouts,
        pop=args.pop,
        keep_boxes=analysis.keep_boxes,
        has_face=analysis.has_face,
        pop_threshold=args.pop_threshold,
    )

    out_path = resolve_output_path(Path(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"quality": args.quality} if out_path.suffix.lower() in (".jpg", ".jpeg") else {}
    collage.save(out_path, **save_kwargs)
    print(f"Saved collage ({collage.width}x{collage.height}) to '{out_path}'.")


# This standard Python idiom means: only run main() when the file is executed
# directly (python collage.py). If another file ever `import`ed this one, main()
# would NOT auto-run — letting them reuse the functions above.
if __name__ == "__main__":
    main()
