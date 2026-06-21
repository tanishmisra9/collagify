# photo-collage

Turn a folder of photos into a single portrait collage. CLI only, no GUI.

Drop any number of images into `pics/`, run the script, get `collage.jpg`.

## How it works

This is a **free-flowing tiling layout**, not a fixed row/column grid.
Cells are not forced to a uniform row height — a tall photo gets a tall
cell, a wide photo gets a wide cell, and the canvas is tiled edge-to-edge
with no gutters by default.

1. Reads every image in `pics/` (jpg, jpeg, png, webp, bmp, tiff).
2. Fixes camera EXIF rotation automatically.
3. Recursively splits the canvas in two ("guillotine cuts"), either side
   by side or stacked. At each split, the images are divided into two
   groups, and the split position is set by how much width/height each
   group's photos naturally "want" based on their own aspect ratios —
   not a fixed 50/50. This repeats until every photo has its own
   rectangle.
4. Because each rectangle is sized to roughly match its photo's actual
   aspect ratio, only a small center-crop is needed to square off the
   edges — nowhere near as much gets cut off as a uniform grid would
   require.
5. Solo branches (a rectangle that ends up with just one photo) get a
   size boost so a stand-out photo can read as a "hero" rather than
   getting cramped into a small cell.
6. Everything is composited onto one canvas and saved.

The split structure is randomized (seedable) so re-running on the same
folder gives a different arrangement each time, unless you pass `--seed`.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# put your photos in pics/, then:
python collage.py
```

This reads everything in `pics/` and writes `collage.jpg` in the project
root, sized 1200×1500 (4:5 portrait) by default, with no gutters.

### Options

```
python collage.py [options]

--input-dir DIR             Folder to read images from (default: pics)
--output PATH                Output file path (default: collage.jpg)
--width N                     Canvas width in px (default: 1200)
--ratio W:H                  Canvas aspect ratio, e.g. 4:5, 3:4, 9:16 (default: 4:5)
--gutter N                    Gap between photos in px (default: 0, edge-to-edge)
--gutter-color HEX           Color shown in gutters, only visible if --gutter > 0
--hero-boost N                 How much extra space solo cells get, 1.0 = none (default: 1.5)
--order {random,name,date}  Placement order (default: random)
--seed N                       Random seed for a reproducible layout
--recursive                    Also search subfolders of --input-dir
--quality N                     JPEG output quality, 1-100 (default: 92)
```

### Examples

Reproducible layout (same arrangement every run):

```bash
python collage.py --seed 42
```

Tall phone-shaped collage:

```bash
python collage.py --ratio 9:16 --output story.jpg
```

Flatten out the hero effect so every photo gets roughly equal weight:

```bash
python collage.py --hero-boost 1.0
```

A thin gutter, if you want some separation between photos:

```bash
python collage.py --gutter 6
```

## Notes

- Works with any number of images — 1 fills the whole canvas, 2 splits
  the canvas in two, larger sets tile recursively.
- Crops are always centered and are normally minor, since cell shape is
  matched to each photo's own aspect ratio before any cropping happens.
  An unusually extreme aspect ratio mixed into a very small set of photos
  can still need a noticeable crop — there's only so much space to go
  around.