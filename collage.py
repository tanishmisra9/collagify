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
3. Use an AI model (rembg) to "cut out" the subject in each      -> subject_cutout
   photo (separate the person/object from the background).
4. Plan an invisible grid ("skeleton") that gives each photo a   -> layout_cells
   cell shaped to fit its subject.
5. Composite (paste) everything onto one canvas:                 -> build_collage
   - oversize + jitter the cells so photos overlap a little,
   - paint photo backgrounds back-to-front WITHOUT covering any
     subject that's already there,
   - lift ONE subject on top with a shadow ("the pop").
6. Save the finished image to disk.                              -> main

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


def parse_args():
    """Read the command-line options the user typed and return them as an object.

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
    p.add_argument("--overlap", type=float, default=1.1,
                    help="How much each photo oversizes its cell so neighbors overlap "
                         "(1.0 = no overlap; default: 1.1)")
    p.add_argument("--scatter", type=float, default=0.15,
                    help="How much photos are jittered off-grid, 0-1 (default: 0.15)")
    p.add_argument("--pop", type=int, default=1,
                    help="Max number of subjects that pop out over neighbors (0 = none; default: 1)")
    p.add_argument("--model", default="birefnet-general-lite",
                    help="rembg segmentation model (default: birefnet-general-lite)")
    p.add_argument("--no-cutout", action="store_true",
                    help="Skip rembg subject cutouts entirely; plain overlap only")
    p.add_argument("--hero-boost", type=float, default=1.5,
                    help="How much extra space single-image branches get, 1.0 = no boost (default: 1.5)")
    p.add_argument("--order", choices=["random", "name", "date"], default="random",
                    help="Order images are placed in (default: random)")
    p.add_argument("--seed", type=int, default=None,
                    help="Random seed, for a reproducible layout")
    p.add_argument("--recursive", action="store_true",
                    help="Also search subdirectories of input-dir")
    p.add_argument("--quality", type=int, default=92,
                    help="JPEG output quality 1-100 (default: 92)")
    return p.parse_args()


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


def plan_layout(specs, i, j, rect_aspect, boost, rng, memo):
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
        # area proportional to its photo count, adjusted by hero_boost (a lone
        # photo gets extra). fa is group A's fraction of the rectangle.
        area_a = na * hero_boost(na, boost)
        area_b = nb * hero_boost(nb, boost)
        fa = area_a / (area_a + area_b)

        # Now recurse for BOTH ways of cutting, and figure out each sub-piece's
        # shape:
        #  - vertical cut ("v"): two side-by-side columns. They share the full
        #    height, and split the width by `fa`, so a column's aspect = parent
        #    aspect * its width fraction.
        #  - horizontal cut ("h"): two stacked rows. They share the full width,
        #    and split the height by `fa`, so a row's aspect = parent aspect /
        #    its height fraction.
        cav, nav = plan_layout(specs, i, i + k, rect_aspect * fa, boost, rng, memo)
        cbv, nbv = plan_layout(specs, i + k, j, rect_aspect * (1 - fa), boost, rng, memo)
        cah, nah = plan_layout(specs, i, i + k, rect_aspect / fa, boost, rng, memo)
        cbh, nbh = plan_layout(specs, i + k, j, rect_aspect / (1 - fa), boost, rng, memo)

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


def cell_spec(img, cut):
    """Describe a photo's *shape preference* for the layout, as
    (target_aspect, narrow_penalty, wide_penalty).

    This is where we tell the layout "please don't cut this person off." The
    layout will fit each photo's cell shape to its `target_aspect`, but no canvas
    can match every photo perfectly, so some cropping is unavoidable. The two
    penalties say which *direction* of mismatch is worse for THIS photo:

      - A cell NARROWER than the photo crops the photo's LEFT/RIGHT edges. In a
        group photo, the outer people's faces live on the left and right — so
        side-cropping is bad. narrow_penalty grows with the subject's WIDTH (sw).
      - A cell WIDER than the photo crops the TOP/BOTTOM. That's usually sky,
        floor, or torsos — less bad. wide_penalty grows with the subject's
        HEIGHT (sh).

    The subject's width/height come from the cutout's bounding box (`cut.bbox`,
    normalized 0-1). We multiply the narrow penalty by 1.6 to lean even harder
    against slicing faces — that's the fix that finally kept a wide two-person
    selfie from getting one person chopped off: it makes such a photo strongly
    prefer a WIDE cell.

    Edge cases:
      - cut is None (no cutout for this photo, e.g. --no-cutout, or rembg failed):
        we have no subject info, so we fall back to symmetric penalties (1.0, 1.0)
        and just match the image's own aspect.
      - max(0.4, ...) keeps a minimum penalty so even a tiny subject still nudges
        the layout (and avoids a zero penalty that the cost search would ignore).
    """
    a = aspect_ratio(img)
    if cut is None:
        return (a, 1.0, 1.0)
    sw = cut.bbox[2] - cut.bbox[0]   # subject width fraction  (right - left)
    sh = cut.bbox[3] - cut.bbox[1]   # subject height fraction (bottom - top)
    return (a, max(0.4, sw * 1.6), max(0.4, sh))


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


def layout_cells(images, width, height, rng, boost, cutouts=None):
    """Produce the final skeleton: a list of cells [(path, img, x0,y0,x1,y1), ...]
    that tile the whole canvas, each shaped to fit its photo's subject.

    Steps:
      - If there's only ONE photo, it simply fills the entire canvas (no slicing
        needed). This is an important edge case — the search below assumes 2+.
      - Build a `spec` (shape preference) for each photo via cell_spec, looking up
        its cutout in `cutouts` (a dict keyed by the file path as text). `cutouts`
        may be None (when --no-cutout); then specs fall back to image aspect.
      - Sort the photos by their target aspect. Sorting groups similar shapes
        next to each other, which the guillotine search can then keep clustered
        (e.g. all the tall photos in one column). plan_layout splits *contiguous*
        ranges, so the sort order matters.
      - "Best-of-N": plan_layout already finds a near-best plan, but its random
        tie-breaking means different runs find different near-best plans. So we
        run it 8 times (each consuming the rng a little differently) and keep the
        one with the lowest cost — the tightest subject fit. This makes the
        *quality* consistent run to run, while a different --seed still explores a
        different set of candidates (so you still get variety). The `{}` passed as
        `memo` is a fresh empty cache for each attempt.
      - Finally, realize_layout turns the winning plan into real pixel rectangles.
    """
    n = len(images)
    if n == 1:
        path, img = images[0]
        return [(path, img, 0, 0, width, height)]

    def spec_of(item):
        path, img = item
        return cell_spec(img, cutouts.get(str(path)) if cutouts else None)

    items = sorted(images, key=lambda it: spec_of(it)[0])  # cluster by image aspect
    specs = [spec_of(it) for it in items]

    best_cost, best_node = None, None
    for _ in range(8):
        cost, node = plan_layout(specs, 0, n, width / height, boost, rng, {})
        if best_cost is None or cost < best_cost:
            best_cost, best_node = cost, node
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


def cover_transform(img, target_w, target_h, bbox=None):
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
    if bbox is None:
        return nw, nh, (nw - target_w) // 2, (nh - target_h) // 2
    # Subject box in scaled-image pixels:
    bx0, by0, bx1, by1 = bbox[0] * nw, bbox[1] * nh, bbox[2] * nw, bbox[3] * nh
    # Horizontal: center the window on the box, then nudge to keep the box inside.
    left = round((bx0 + bx1) / 2 - target_w / 2)
    left = min(max(left, round(bx1) - target_w), round(bx0))   # keep bbox inside if it fits
    left = max(0, min(nw - target_w, left))                    # stay within the image
    # Vertical: same idea.
    top = round((by0 + by1) / 2 - target_h / 2)
    top = min(max(top, round(by1) - target_h), round(by0))
    top = max(0, min(nh - target_h, top))
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


def build_collage(images, width, ratio, boost, rng, overlap, scatter, cutouts, pop):
    """Assemble the final collage image and return it.

    This is the big one. Inputs: the loaded `images`, the canvas `width` and shape
    `ratio`, the hero `boost`, the seeded `rng`, the `overlap`/`scatter` look
    knobs, the `cutouts` dict (path -> Cutout or None; may be None entirely), and
    `pop` (how many subjects may be lifted on top).

    High-level plan:
      A. Compute the canvas height from width and ratio; make a dimmed, blurred
         backdrop so any tiny gaps read as soft depth, not empty bars.
      B. Lay out the skeleton, then oversize+jitter each cell into an overlapping
         "placement."
      C. Precompute, for each photo, three maps: its rectangle, where its subject
         shows, and the subject's full reach if lifted on top.
      D. Pick the ONE subject to feature ("pop"), carefully avoiding covering
         anyone else.
      E. Pass 1: paint photo backgrounds back-to-front, but never over a subject
         already placed (this is the no-one-gets-covered guarantee).
      F. Pass 2: lift the featured subject on top with a soft shadow.
    """
    # --- A. Canvas + backdrop ----------------------------------------------
    height = round(width / ratio)
    cells = layout_cells(images, width, height, rng, boost, cutouts)

    # Backdrop: take one photo, blow it up to cover the canvas, blur it heavily,
    # and darken it (multiply every pixel by 0.55). If jitter ever leaves a sliver
    # uncovered, this soft dark blur shows through instead of a hard empty bar.
    canvas = cover_crop(images[0][1], width, height)
    canvas = canvas.filter(ImageFilter.GaussianBlur(max(width, height) // 12))
    canvas = canvas.point(lambda p: int(p * 0.55))

    # --- B. Turn each cell into an overlapping, slightly-jittered placement --
    # Each "placement" is (path, img, px0, py0, pw, ph): top-left corner + size.
    placements = []
    for path, img, x0, y0, x1, y1 in cells:
        cw, ch = x1 - x0, y1 - y0                 # the skeleton cell's size
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2     # its center
        # Oversize the cell by `overlap` (e.g. 1.1 = 10% bigger) so neighbors
        # overlap a little instead of butting edge-to-edge.
        pw, ph = max(1, round(cw * overlap)), max(1, round(ch * overlap))
        # Jitter: nudge the center by a small random amount so the result looks
        # hand-placed, not grid-perfect. The nudge is at most half the extra size
        # times `scatter`, and uses rng so it's reproducible per seed.
        # (rng.random()*2 - 1) is a random number in [-1, 1].
        jx = (rng.random() * 2 - 1) * (pw - cw) * 0.5 * scatter
        jy = (rng.random() * 2 - 1) * (ph - ch) * 0.5 * scatter
        px0 = round(cx + jx - pw / 2)
        py0 = round(cy + jy - ph / 2)
        placements.append((path, img, px0, py0, pw, ph))

    # Small helper: get the Cutout for placement #idx (or None).
    def cut_of(idx):
        return cutouts.get(str(placements[idx][0])) if cutouts else None

    # Paint order, back to front: biggest cells first (so they're at the back) and
    # smaller photos painted later (on top), so a small photo isn't buried under a
    # big one. `order` is a list of placement indices, largest area first.
    order = sorted(range(len(placements)),
                   key=lambda i: placements[i][4] * placements[i][5], reverse=True)

    # --- C. Precompute the maps for each photo -----------------------------
    #   own[idx]     : the photo's rectangle (a True/False map).
    #   face_ft[idx] : where this subject actually SHOWS = its silhouette clipped
    #                  to its own rectangle. Used to protect it and to test if it
    #                  sits near a seam.
    #   pop_ft[idx]  : the subject's FULL silhouette if lifted on top — this can
    #                  reach past the rectangle (a lifted cutout spills outward).
    #                  Used to test what a "pop" would cover.
    own, face_ft, pop_ft = {}, {}, {}
    for idx, (path, img, px0, py0, pw, ph) in enumerate(placements):
        own[idx] = _rect_mask(px0, py0, pw, ph, width, height)
        cut = cut_of(idx)
        if cut is not None:
            fnw, fnh, fl, ftop = cover_transform(img, pw, ph, cut.bbox)
            full = _footprint(cut.mask, fnw, fnh, px0 - fl, py0 - ftop, width, height)
            # Why two versions: clipping to the rectangle (face_ft) tells us where
            # the subject is genuinely painted. Protecting pixels OUTSIDE the
            # rectangle would punch holes no photo fills, exposing the backdrop —
            # so face_ft (clipped) is what we protect. pop_ft (unclipped) is the
            # true reach of a lifted cutout, used for the collision check.
            pop_ft[idx] = full
            face_ft[idx] = full & own[idx]

    # --- D. Choose the single featured subject to "pop" --------------------
    # We want a clean, nicely-sized subject that sits where it overlaps a neighbor
    # (so lifting it reads as leaning over that neighbor) — but that won't cover
    # anyone else's face. We score every candidate and pick the best valid one.
    selected = []
    if cutouts and pop > 0:
        scored = []
        for idx in face_ft:
            fa = face_ft[idx]
            fsum = int(fa.sum())      # how many pixels this subject covers
            if fsum == 0:
                continue              # no visible subject; can't pop it

            # "seam": what fraction of this subject sits inside a neighbor's
            # rectangle (i.e. in an overlap zone). Higher = lifting it creates a
            # nice in-front-of-neighbor depth effect rather than floating in place.
            neighbors = np.zeros((height, width), dtype=bool)
            for j in own:
                if j != idx:
                    neighbors |= own[j]
            seam = (fa & neighbors).sum() / fsum

            # "collision": the worst fraction of ANY single OTHER subject this pop
            # would cover, using the pop's FULL reach (pop_ft). We measure per
            # other-subject (not as a share of the pop) on purpose: a huge cutout
            # like a race car covering a small face is a tiny share of the car but
            # a big share of the face — this catches that.
            collision = 0.0
            for j in face_ft:
                if j != idx and face_ft[j].any():
                    covered = int((pop_ft[idx] & face_ft[j]).sum()) / int(face_ft[j].sum())
                    collision = max(collision, covered)

            # Score combines: how clean the cutout is (isolation), how well-sized
            # the subject is (size_fit peaks when it fills ~35% of its photo — not
            # a tiny speck, not the whole frame), and how much it leans over a
            # neighbor (seam). Higher score = better pop candidate.
            cut = cut_of(idx)
            size_fit = max(0.0, 1.0 - abs(cut.coverage - 0.35) / 0.35)
            score = cut.isolation * size_fit * min(1.0, seam * 4.0)
            scored.append((score, collision, idx))

        # Take the best-scoring candidates, but skip any that would cover more than
        # ~6% of another subject. Stop once we've chosen `pop` of them (default 1).
        scored.sort(reverse=True)             # highest score first
        for score, collision, idx in scored:
            if len(selected) >= pop or score <= 0:
                break
            if collision <= 0.06:
                selected.append(idx)

    # Friendly log line so you can see which photo (if any) got the pop.
    if cutouts:
        names = [Path(placements[i][0]).name for i in selected]
        print(f"  popped {len(names)}: {', '.join(names) if names else 'none'}")

    # --- E. Pass 1: paint backgrounds, never covering an existing subject ---
    # We keep a running `protected` map of every subject painted so far. When we
    # paste the next photo, its paste-mask is fully opaque EXCEPT on protected
    # pixels (set to 0 there), so the new photo paints around already-placed
    # subjects instead of over them. Because the subject's own photo is underneath
    # those pixels, the result is the subject peeking through — never a gap.
    protected = np.zeros((height, width), dtype=bool)
    for idx in order:
        path, img, px0, py0, pw, ph = placements[idx]
        cut = cut_of(idx)
        bg = subject_cover_crop(img, pw, ph, cut.bbox if cut else None)
        # Start with a fully-opaque (255) paste mask the size of this photo...
        mask = np.full((ph, pw), 255, dtype=np.uint8)
        # ...then punch holes (0) wherever a lower subject is protected. We only
        # touch the on-canvas overlap region (clamped by cx0..cy1) so off-edge
        # photos don't error.
        cx0, cy0 = max(0, px0), max(0, py0)
        cx1, cy1 = min(width, px0 + pw), min(height, py0 + ph)
        if cx1 > cx0 and cy1 > cy0:
            mask[cy0 - py0:cy1 - py0, cx0 - px0:cx1 - px0][protected[cy0:cy1, cx0:cx1]] = 0
        canvas.paste(bg, (px0, py0), Image.fromarray(mask, "L"))
        # Now this photo's own subject becomes protected from anything painted next.
        if idx in face_ft:
            protected |= face_ft[idx]

    # --- F. Pass 2: lift the featured subject on top with a soft shadow -----
    for idx in order:
        if idx not in selected:
            continue
        path, img, px0, py0, pw, ph = placements[idx]
        cut = cut_of(idx)
        # Use the SAME transform as this photo's background so the lifted cutout
        # lands exactly over where the subject already is (no double image).
        fnw, fnh, fl, ftop = cover_transform(img, pw, ph, cut.bbox)
        sub = cut.rgba.resize((fnw, fnh), Image.LANCZOS)
        x, y = px0 - fl, py0 - ftop
        # Shadow: a black copy shaped like the subject's alpha, blurred and made
        # faint (40% opacity), offset down-right by a few pixels. Pasting it
        # first, then the sharp subject on top, makes the subject look lifted off
        # the photo behind it.
        shadow = Image.new("RGBA", sub.size, (0, 0, 0, 0))
        shadow.putalpha(sub.getchannel("A").filter(ImageFilter.GaussianBlur(10))
                        .point(lambda p: int(p * 0.4)))
        canvas.paste(shadow, (x + 7, y + 9), shadow)
        canvas.paste(sub, (x, y), sub)

    return canvas


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
      7. Unless --no-cutout: load the rembg model ONCE (new_session) and run the
         cutout on every photo, storing results in `cutouts` keyed by file path.
         If rembg isn't installed or the model fails to load, we warn and fall
         back to cutouts = None (the collage still works, just with no AI cutouts).
      8. Build the collage.
      9. Pick a non-clobbering output path, make sure its folder exists
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

    # Run the AI cutout on each photo (unless disabled). One shared `session`
    # avoids reloading the ~200MB model per image. Any failure here degrades
    # gracefully to "no cutouts" rather than stopping the program.
    cutouts = None
    if not args.no_cutout:
        try:
            from rembg import new_session
            session = new_session(args.model)
            cutouts = {}
            for path, img in images:
                cut = subject_cutout(img, session)
                cutouts[str(path)] = cut
                print(f"  cutout {path.name}: {'ok' if cut is not None else 'skipped'}")
        except Exception as e:
            print(f"  ! cutouts unavailable ({e}); using plain overlap only",
                  file=sys.stderr)
            cutouts = None

    collage = build_collage(
        images=images,
        width=args.width,
        ratio=ratio,
        boost=args.hero_boost,
        rng=rng,
        overlap=args.overlap,
        scatter=args.scatter,
        cutouts=cutouts,
        pop=args.pop,
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
