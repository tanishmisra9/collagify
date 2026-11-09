"""Runnable self-check for the plan/render split and the GUI session.

    .venv/bin/python test_gui_core.py

Plain asserts, no framework. Uses the photos in pics/ (and the rembg model for
the cutout cases, so the first run may download it).
"""
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import collage
from gui import schema
from gui.session import Session

PICS = sorted(Path("pics").glob("*.jpg"))
assert len(PICS) >= 2, "put 2+ photos in pics/"


def opts(**kw):
    base = dict(no_color=False, color_strength=0.35, no_cutout=False, model="birefnet-general-lite", no_faces=False)
    return SimpleNamespace(**{**base, **kw})


def make_plan(analysis, seed=3, pop=3, thr=0.0):
    rng = random.Random(seed)
    imgs = collage.order_images(list(analysis.images), "name", rng)
    return collage.plan_collage(imgs, 540, 9 / 16, 1.5, rng, 1.15, 0.5, analysis.cutouts, pop,
                                analysis.keep_boxes, analysis.has_face, thr)


def arr(img):
    return np.asarray(img, dtype=np.int16)


def top_idx(plan):
    return plan.order[-1]   # painted last = fully visible (no-cutout case)


def test_schema():
    d = collage.build_parser().parse_args([])
    assert schema.defaults()["overlap"] == d.overlap and schema.defaults()["width"] == d.width
    assert schema.normalize({"overlap": 99})["overlap"] == 1.5            # clamps
    assert schema.normalize({"pop": 2.6})["pop"] == 3                      # int slider rounds
    for bad in ({"ratio": "7:7"}, {"overlap": "abc"}, {"model": "../x"}):
        try:
            schema.normalize(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    assert schema.normalize({})["seed"] is None


def test_no_cutout(analysis):
    plan = make_plan(analysis)
    base = arr(collage.render_plan(plan))
    assert base.shape == (960, 540, 3)
    assert (arr(collage.render_plan(plan)) == base).all(), "cached re-render differs"
    t = top_idx(plan)
    _, _, px0, py0, pw, ph = plan.placements[t]
    sh = arr(collage.render_plan(plan, offsets={t: (20, 0, 0, 0)}))
    ys, ye = max(py0, 0) + 5, min(py0 + ph, 960) - 5
    xs, xe = max(px0, 0) + 5, min(px0 + pw, 540) - 45
    assert xe > xs and ye > ys
    assert (sh[ys:ye, xs + 20:xe + 20] == base[ys:ye, xs:xe]).all(), "dx did not move the frame by 20px"
    # moving then moving back must equal the cold render (cache not stale)
    collage.render_plan(plan, offsets={t: (-30, 15, 0, 0)})
    plan.cache.clear()
    assert (arr(collage.render_plan(plan)) == base).all()
    # preview scale keeps geometry proportions
    assert collage.render_plan(plan, 0.5).size == (270, 480)


def test_pan(analysis, label):
    plan = make_plan(analysis, pop=0)
    base = arr(collage.render_plan(plan))
    moved = False
    for idx in range(len(plan.placements)):
        (x0, x1), (y0, y1) = collage.pan_limits(plan, idx)
        if x1 > 0 or y1 > 0 or x0 < 0 or y0 < 0:
            pan = (min(x1, 6) if x1 > 0 else max(x0, -6), min(y1, 6) if y1 > 0 else max(y0, -6))
            out = arr(collage.render_plan(plan, offsets={idx: (0, 0) + pan}))
            assert (out != base).any(), f"{label}: pan {pan} on photo {idx} changed nothing"
            moved = True
            # a pan past the limit must be clamped, never expose an empty edge
            huge = collage.render_plan(plan, offsets={idx: (0, 0, 10**6, 10**6)})
            assert huge.size == (540, 960)
    assert moved, f"{label}: no photo had pan slack"
    for idx in range(len(plan.placements)):   # pan limits must be a valid interval containing 0
        (x0, x1), (y0, y1) = collage.pan_limits(plan, idx)
        assert x0 <= 0 <= x1 and y0 <= 0 <= y1


def test_pan_keeps_pop_aligned(analysis):
    plan = make_plan(analysis, pop=6, thr=0.0)
    if not plan.selected:
        print("  (no pops available in pics/; alignment case skipped)")
        return
    idx = plan.selected[0]
    pan = (collage.pan_limits(plan, idx)[0][1], collage.pan_limits(plan, idx)[1][1])
    geo = collage._geometry(plan, {idx: (0, 0) + pan}, 1.0)
    _, face, pop, lift = collage._footprints(plan, geo, plan.width, plan.height)
    ys, xs = np.nonzero(pop[idx])
    lnw, lnh, lx, ly = lift[idx]
    assert lx - 2 <= xs.min() and xs.max() <= lx + lnw + 2, "lifted cutout not where lift_place says"
    assert face[idx].any()


def test_single_photo():
    a = collage.analyze(collage.load_images(PICS[:1], "name", None), opts(no_cutout=True, no_faces=True))
    rng = random.Random(1)
    plan = collage.plan_collage(list(a.images), 540, 9 / 16, 1.5, rng, 1.1, 0.3, None, 1, a.keep_boxes, a.has_face)
    assert collage.render_plan(plan).size == (540, 960)


def test_session(analysis):
    s = Session(tempfile.mkdtemp())
    try:
        s.plan = make_plan(analysis, pop=0)
        try:
            s.set_offset(99, 0, 0, 0, 0)
            raise AssertionError("bad id accepted")
        except KeyError:
            pass
        v = s.set_offset(0, 25, -10, 0, 0)
        assert v[:2] == (25, -10) and s.undo
        far = s.set_offset(0, 10**6, 10**6, 0, 0)         # clamped to keep >=30% on canvas
        _, _, px0, py0, pw, ph = s.plan.placements[0]
        assert far[0] <= s.plan.width - 0.3 * pw - px0 + 1 and far[1] <= s.plan.height - 0.3 * ph - py0 + 1
        s.step(-1); s.step(-1)
        assert s.offsets == {} and not s.undo and len(s.redo) == 2
        s.step(+1)
        assert s.offsets[0][:2] == (25, -10)
        s.set_offset(0, 25, -10, 0, 0)                      # no-op must not push undo
        assert len(s.redo) == 1, "a no-op edit must keep redo intact"
        s.set_offset(0, 5, 5, 0, 0)
        assert not s.redo
        s.reset(0)
        assert s.offsets == {}
        assert s.jpeg(0.3, (0, 40, 40, 0, 0))[:2] == b"\xff\xd8"
        assert s.offsets == {}, "preview override must not mutate state"
        out = s.export(90)
        assert out.exists() and out.parent == s.out_dir
        try:
            s.add_upload("x.heic", b"1")
            raise AssertionError("heic accepted")
        except ValueError as e:
            assert "JPEG" in str(e)
        try:
            s.add_upload("../../evil.jpg", b"not an image")
            raise AssertionError("garbage accepted")
        except ValueError:
            pass
        assert not any(s.work.glob("*evil*")), "rejected upload left a file behind"
        big = Path(tempfile.mkdtemp())
        for i in range(collage.MAX_PHOTOS + 1):
            (big / f"{i}.jpg").write_bytes(PICS[0].read_bytes())
        try:
            s.add_folder(big)
            raise AssertionError("oversized folder accepted")
        except ValueError:
            assert s.photos == [], "a rejected folder must not leave photos behind"
    finally:
        s.close()


if __name__ == "__main__":
    test_schema()
    imgs = collage.load_images(PICS, "name", None)
    plain = collage.analyze(imgs, opts(no_cutout=True))
    test_no_cutout(plain)
    test_pan(plain, "no-cutout")
    full = collage.analyze(collage.load_images(PICS, "name", None), opts())
    test_pan(full, "cutout+faces")
    test_pan_keeps_pop_aligned(full)
    test_single_photo()
    test_session(plain)
    print("all checks passed")
