# Collagify

Turn a folder of photos into a polished, layered collage with one command. A subject-aware layout keeps every face fully visible (never cropped or covered), and one local AI cutout pops out on top with a soft drop shadow. Runs fully local, no API keys.

Drop your photos in `pics/`, run the script, get a collage in `output/`.

## Features

- **Subject-aware layout.** Each photo gets a cell sized to fit its subject, so a wide two-person photo is never sliced down to one face.
- **Layered, overlapping look.** Photos overlap slightly with clean hard edges, like printed pictures stacked on a table.
- **No subject ever covered.** Overlaps are composited so an overlapping photo reveals the subject beneath it instead of hiding it.
- **One tasteful 3D pop.** A single clean subject is cut from its background and lifted on top with a soft shadow, picked so it never covers another face.
- **Fully local.** Background removal runs on your machine via [rembg](https://github.com/danielgatis/rembg). No accounts, no API keys, no uploads.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies are Pillow and `rembg[cpu]` (which pulls in `onnxruntime`). The first run downloads the segmentation model (~224 MB) once, then works offline. Use `--no-cutout` to skip the AI step entirely.

## Usage

```bash
# put your photos in pics/, then:
python collage.py
```

This reads everything in `pics/` and writes `output/collage.jpg`, sized 1080x1920 (9:16, Instagram Stories) by default. Re-running adds `collage2.jpg`, `collage3.jpg`, and so on rather than overwriting.

### Options

| Option | Default | Description |
| --- | --- | --- |
| `--input-dir DIR` | `pics` | Folder to read images from. |
| `--output PATH` | `output/collage.jpg` | Output file path (auto-increments if it exists). |
| `--width N` | `1080` | Canvas width in pixels. |
| `--ratio W:H` | `9:16` | Canvas aspect ratio, e.g. `4:5`, `3:4`, `9:16`. |
| `--overlap F` | `1.1` | How much each photo oversizes its cell so neighbors overlap (`1.0` = no overlap). |
| `--scatter F` | `0.15` | How much photos are jittered off the grid, `0`-`1`. |
| `--pop N` | `1` | Max number of subjects lifted out over neighbors (`0` = none). |
| `--model NAME` | `birefnet-general-lite` | rembg segmentation model. |
| `--no-cutout` | off | Skip the AI cutouts entirely; plain overlapping collage. |
| `--hero-boost F` | `1.5` | Extra space a lone photo gets (`1.0` = none). |
| `--order {random,name,date}` | `random` | Order photos are placed in. |
| `--seed N` | random | Random seed, for a reproducible result. |
| `--recursive` | off | Also search subfolders of the input directory. |
| `--quality N` | `92` | JPEG output quality, `1`-`100`. |

### Examples

Reproducible result (same arrangement every run):

```bash
python collage.py --seed 42
```

A looser, more scattered arrangement:

```bash
python collage.py --scatter 0.3 --overlap 1.2
```

A clean square collage with no AI cutouts (fastest):

```bash
python collage.py --ratio 1:1 --no-cutout
```

## How it works

1. **Cutouts.** rembg separates the subject from the background in each photo, and the mask is cleaned up (largest blob kept, edges softened).
2. **Layout.** A recursive "guillotine" search slices the canvas into one cell per photo, choosing the slicing that best fits each cell's shape to its subject. It is asymmetric on purpose: cutting a photo's sides (where faces sit) is penalized harder than cutting top or bottom.
3. **Composite.** Cells are oversized and jittered so photos overlap, then painted back to front. Each photo paints around any subject already placed, so no face is ever covered. Finally one subject is lifted on top with a shadow.

## Notes

- Works with any number of photos: one fills the canvas, two stack, larger sets tile.
- Runs on CPU. With the default model, expect a few seconds per photo for the cutout step.
- Everything stays on your machine. Nothing is uploaded.
