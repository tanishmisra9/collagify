# Collagify

Turn a folder of photos into a polished, layered collage with one command. A subject-aware layout keeps every face fully visible (never cropped or covered), and one local AI cutout pops out on top with a soft drop shadow. Runs fully local, no API keys.

Drop your photos in `pics/`, run the script, get a collage in `output/`.

## Features

- **Face-aware layout & cropping.** Faces are detected (InsightFace SCRFD) and drive both how cells are shaped and how each photo is cropped, so faces are never sliced at a seam and a wide group photo gets a wide cell. Background-crowd faces are filtered out using the cutout, so framing stays on the people the photo is about.
- **Color cohesion.** All photos are graded toward one shared look, so a set shot in mixed lighting feels like one cohesive piece instead of a random pile.
- **Layered, overlapping look.** Photos overlap slightly with clean hard edges, like printed pictures stacked on a table.
- **No subject ever covered.** Overlaps are composited so an overlapping photo reveals the subject beneath it instead of hiding it.
- **Tasteful 3D pops.** Clean subjects are cut from their background and lifted on top, scaled up slightly with a drop shadow so they genuinely stand off the page. It's quality-gated, not a fixed count: a set with no standout pops nothing, a set with several great subjects pops a few (only ever lifting people, never an object/poster/screenshot, never covering a face, never clustering).
- **Fully local.** Background removal and face detection run on your machine ([rembg](https://github.com/danielgatis/rembg), [InsightFace](https://github.com/deepinsight/insightface)). No accounts, no API keys, no uploads.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies are Pillow, `rembg[cpu]`, and `insightface` (which also brings in `onnxruntime` and OpenCV). The first run downloads the segmentation model (~224 MB) and a small face-detection model pack once, then works offline. Use `--no-cutout`, `--no-faces`, or `--no-color` to skip individual AI steps.

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
| `--overlap F` | `1.18` | How much each photo oversizes its cell so neighbors overlap (`1.0` = no overlap). |
| `--scatter F` | `0.22` | How much photos are jittered off the grid, `0`-`1`. |
| `--pop N` | `3` | Maximum subjects lifted out over neighbors. Quality-gated, so it pops fewer or none (`0` = never). |
| `--pop-threshold F` | `0.25` | Minimum "tastefulness" score for a subject to pop; higher = more selective. |
| `--color-strength F` | `0.35` | How strongly photos are graded toward a shared look, `0`-`1`. |
| `--model NAME` | `birefnet-general-lite` | rembg segmentation model. |
| `--no-cutout` | off | Skip the AI cutouts entirely; plain overlapping collage. |
| `--no-faces` | off | Skip face detection; frame on the rembg subject only. |
| `--no-color` | off | Skip color cohesion grading. |
| `--hero-boost F` | `1.5` | Extra space a lone photo gets (`1.0` = none). |
| `--order {random,name,date}` | `random` | Order photos are placed in. |
| `--seed N` | random | Random seed, for a reproducible result. |
| `--recursive` | off | Also search subfolders of the input directory. |
| `--max-photos N` | `6` | Reject inputs with more than this many images (a denser collage gets cramped). |
| `--quality N` | `100` | JPEG output quality, `1`-`100`. |

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

1. **Color.** Every photo is graded toward the set's average tone (a perceptual LAB match), blended subtly so the collage feels cohesive without looking heavily filtered.
2. **Cutouts.** rembg separates the subject from the background in each photo, and the mask is cleaned up (largest blob kept, edges softened).
3. **Faces.** InsightFace finds the faces, drops distant background-crowd faces (by size and by whether they sit on the cutout subject), and builds a "must-stay-in-frame" region per photo.
4. **Layout.** A recursive "guillotine" search slices the canvas into one cell per photo, choosing the slicing that best fits each cell's shape to that face region. It is asymmetric on purpose: cutting a photo's sides (where faces sit) is penalized harder than cutting top or bottom.
5. **Composite.** Cells are oversized and jittered so photos overlap, then painted back to front. Each photo paints around any subject already placed, so no face is ever covered. Finally, subjects that score highly enough (clean, well-sized, sitting over an overlap, ideally a person) are lifted on top with a shadow — zero, one, or a few depending on how many genuinely earn it.

## Notes

- Works with 1 to 6 photos (the `--max-photos` cap): one fills the canvas, two stack, a handful tile. More than that gets cramped in the frame, so it is rejected by default (raise `--max-photos` to override).
- Runs on CPU. With the default model, expect a few seconds per photo for the cutout step.
- Everything stays on your machine. Nothing is uploaded.

## GUI

```bash
python gui.py          # opens http://127.0.0.1:8765 in your browser
```

A local web app (stdlib only, no extra dependencies; it binds to localhost and nothing leaves your machine).

- Add 1–6 photos (drop them in, pick them, or load a folder such as `pics/`), then **Generate**. **Shuffle** picks a new seed.
- Every setting is in the right-hand inspector. Layout-only changes (overlap, scatter, ratio, seed, pops…) re-apply in about a second; settings marked *Slower* (color match, cutouts, faces, model) re-run the AI steps.
- **Fine-tune after generating:** drag a photo to nudge its frame, **Shift+drag** to pan the picture inside its frame, arrow keys nudge 1px (Shift = 10px, Alt+arrows pan). **Alt+click** picks the photo underneath. Undo/redo with ⌘Z / ⇧⌘Z. Overlaps still never cover a subject while you drag.
- Shortcuts: `G` generate · `S` shuffle · `E` export · `[` `]` cycle photos · `Esc` deselect.
- **Export** writes to `output/` with the same auto-incrementing names as the CLI.

Note: generating a new layout clears nudges. `python test_gui_core.py` and `python test_gui_http.py` run the self-checks. Fonts (Instrument Sans, Space Mono, SIL OFL) are bundled in `gui/static/fonts/`.
