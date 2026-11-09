"""All GUI state and logic, with no HTTP in it: photos, the cached analysis, the
current Plan, per-photo offsets, undo/redo, and the background generate job."""
import copy
import io
import os
import random
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageOps

import collage
from gui import schema

THUMB = 240
MAX_UPLOAD = 40 * 1024 * 1024
MIN_VISIBLE = 0.3   # a dragged frame keeps at least this fraction on the canvas


class Busy(Exception):
    pass


class Session:
    def __init__(self, out_dir):
        self.work = Path(tempfile.mkdtemp(prefix="collagify-"))
        self.out_dir = Path(out_dir)
        self.photos = []                 # {id, path, name, w, h}
        self.thumbs = {}                 # id -> jpeg bytes
        self.settings = schema.defaults()
        self.plan = None
        self.plan_photos = []            # path names in placement order
        self.offsets = {}                # idx -> (dx, dy, panx, pany)
        self.undo, self.redo = [], []
        self._last = (None, 0.0)         # (idx, time) of the last committed nudge
        self.rev = 0                     # bumps whenever the picture changes
        self.exports = {}                # name -> path
        self._analysis = None            # (key, Analysis)
        self._job = None
        self._render_lock = threading.Lock()
        self._lock = threading.Lock()
        self._photo_lock = threading.Lock()   # photo list + cap: uploads run on concurrent threads
        self.status = dict(state="idle", message="", progress=0.0, error=None)

    def close(self):
        shutil.rmtree(self.work, ignore_errors=True)

    # ---- photos ---------------------------------------------------------
    def _check_room(self):
        if len(self.photos) >= collage.MAX_PHOTOS:
            raise ValueError(f"A collage holds at most {collage.MAX_PHOTOS} photos. Remove one first.")

    def _register(self, path, name):
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                w, h = im.size
                t = im.copy()
                t.thumbnail((THUMB, THUMB))
        except Exception:
            raise ValueError(f"{name} could not be read as an image.")
        pid = uuid.uuid4().hex[:8]
        buf = io.BytesIO()
        t.save(buf, "JPEG", quality=85)
        self.thumbs[pid] = buf.getvalue()
        photo = dict(id=pid, path=str(path), name=name, w=w, h=h)
        self.photos.append(photo)
        return photo

    def add_upload(self, name, data, modified_ms=None):
        """Store an upload. It lives in its own folder so path.name is the real filename
        (the name/date photo orders sort on it); `modified_ms` is the browser's lastModified."""
        name = Path(name).name or "photo"
        if Path(name).suffix.lower() not in collage.SUPPORTED_EXTS:
            hint = " Export it as JPEG first." if Path(name).suffix.lower() in (".heic", ".heif") else ""
            raise ValueError(f"{name}: unsupported type.{hint} Use JPG, PNG, WebP, BMP or TIFF.")
        with self._photo_lock:
            self._check_room()
            folder = self.work / uuid.uuid4().hex[:8]
            folder.mkdir()
            dest = folder / name
            dest.write_bytes(data)
            try:
                t = float(modified_ms) / 1000
                if 0 < t < 4e9:
                    os.utime(dest, (t, t))
            except (TypeError, ValueError):
                pass                      # no usable date: keep the upload time
            try:
                return self._register(dest, name)
            except ValueError:
                shutil.rmtree(folder, ignore_errors=True)
                raise

    def add_folder(self, folder):
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"'{folder}' is not a folder.")
        files = collage.find_images(folder, False)
        if not files:
            raise ValueError(f"No supported images in '{folder}'.")
        with self._photo_lock:
            room = collage.MAX_PHOTOS - len(self.photos)
            if len(files) > room:   # all-or-nothing: never leave a half-loaded folder behind
                raise ValueError(f"'{folder.name}' has {len(files)} images but there is room for {room}. "
                                 f"A collage holds at most {collage.MAX_PHOTOS}.")
            added = []
            try:
                for f in files:
                    added.append(self._register(f, f.name))
            except ValueError:
                self.photos = [p for p in self.photos if p not in added]
                raise
            return added

    def remove(self, pid):
        with self._photo_lock:
            before = len(self.photos)
            self.photos = [p for p in self.photos if p["id"] != pid]
            if len(self.photos) == before:
                raise KeyError(pid)
            self.thumbs.pop(pid, None)

    # ---- generate -------------------------------------------------------
    def start_generate(self, raw):
        settings = schema.normalize(raw)          # ValueError -> 400
        with self._photo_lock:
            paths = [Path(p["path"]) for p in self.photos]   # snapshot: uploads during a run must not leak in
        if not paths:
            raise ValueError("Add at least one photo first.")
        if len(paths) > collage.MAX_PHOTOS:
            raise ValueError(f"A collage holds at most {collage.MAX_PHOTOS} photos.")
        with self._lock:
            if self._job and self._job.is_alive():
                raise Busy()
            self.status = dict(state="running", message="Starting", progress=0.0, error=None)
            self._job = threading.Thread(target=self._run, args=(settings, paths), daemon=True)
            self._job.start()

    def _progress(self, msg, frac):
        self.status.update(message=msg, progress=round(frac * 0.9, 3))

    def _run(self, s, paths):
        try:
            if s["seed"] is None:
                s["seed"] = random.randrange(1, 10**6)
            ratio = collage.parse_ratio(s["ratio"])
            opts = SimpleNamespace(no_color=s["color_strength"] <= 0, color_strength=s["color_strength"],
                                   no_cutout=not s["cutouts"], model=s["model"], no_faces=not s["faces"])
            key = (tuple((str(p), p.stat().st_mtime_ns) for p in paths),
                   opts.color_strength, opts.no_cutout, opts.model, opts.no_faces)
            warning = None
            if self._analysis and self._analysis[0] == key:
                analysis = self._analysis[1]
            else:
                images = collage.load_images(paths, "name", None)
                if not images:
                    raise ValueError("None of the photos could be loaded.")
                collage.retry_face_detector()
                analysis = collage.analyze(images, opts, self._progress)
                # A degraded analysis (a model that would not load) is shown but never cached,
                # so the next Generate retries instead of silently reusing it.
                missing = []
                if not opts.no_cutout and analysis.cutouts is None:
                    missing.append("subject cutouts")
                if not opts.no_faces and collage.face_detector_failed():
                    missing.append("face detection")
                if missing:
                    warning = (f"{' and '.join(missing).capitalize()} unavailable, so this layout is less "
                               "accurate. Check the server log, then Generate again to retry.")
                    self._analysis = None
                else:
                    self._analysis = (key, analysis)
            self.status.update(message="Laying out", progress=0.93)
            rng = random.Random(s["seed"])
            images = collage.order_images(list(analysis.images), s["order"], rng)
            plan = collage.plan_collage(
                images, s["width"], ratio, s["hero_boost"], rng, s["overlap"], s["scatter"],
                analysis.cutouts, s["pop"], analysis.keep_boxes, analysis.has_face, s["pop_threshold"])
            with self._render_lock:
                self.plan, self.settings = plan, s
                self.plan_photos = [Path(pl[0]).name for pl in plan.placements]
                self.offsets, self.undo, self.redo = {}, [], []
                self.rev += 1
            self.status = dict(state="done", message="Done", progress=1.0, error=None, warning=warning)
        except Exception as e:  # surfaced to the UI; the job thread must not die silently
            self.status = dict(state="error", message="", progress=0.0, error=str(e))

    # ---- offsets --------------------------------------------------------
    def _clamp(self, idx, dx, dy, panx, pany):
        plan = self.plan
        _, _, px0, py0, pw, ph = plan.placements[idx]
        dx = max(-px0 - (1 - MIN_VISIBLE) * pw, min(plan.width - MIN_VISIBLE * pw - px0, dx))
        dy = max(-py0 - (1 - MIN_VISIBLE) * ph, min(plan.height - MIN_VISIBLE * ph - py0, dy))
        (x0, x1), (y0, y1) = collage.pan_limits(plan, idx)
        return (round(dx), round(dy), max(x0, min(x1, round(panx))), max(y0, min(y1, round(pany))))

    def _need_plan(self, idx=None):
        if self.plan is None:
            raise ValueError("Generate a collage first.")
        if idx is not None and not 0 <= idx < len(self.plan.placements):
            raise KeyError(idx)

    def set_offset(self, idx, dx, dy, panx, pany, merge=False):
        """Commit one photo's offsets (one undo step). `merge` folds rapid repeats on
        the same photo (a held arrow key) into the previous step. Returns the clamped value."""
        self._need_plan(idx)
        val = self._clamp(idx, dx, dy, panx, pany)
        with self._render_lock:
            if self.offsets.get(idx, (0, 0, 0, 0)) != val:
                now = time.monotonic()
                if not (merge and self.undo and self._last[0] == idx and now - self._last[1] < 0.8):
                    self.undo.append(copy.deepcopy(self.offsets))
                self._last = (idx, now)
                self.redo.clear()
                self.offsets[idx] = val
                self.rev += 1
        return val

    def reset(self, idx=None):
        self._need_plan(idx)
        with self._render_lock:
            if idx is None and not self.offsets or idx is not None and idx not in self.offsets:
                return
            self.undo.append(copy.deepcopy(self.offsets))
            self.redo.clear()
            self._last = (None, 0.0)
            if idx is None:
                self.offsets = {}
            else:
                self.offsets.pop(idx)
            self.rev += 1

    def step(self, direction):
        """Undo (direction -1) or redo (+1)."""
        src, dst = (self.undo, self.redo) if direction < 0 else (self.redo, self.undo)
        with self._render_lock:
            if not src:
                return
            dst.append(copy.deepcopy(self.offsets))
            self.offsets = src.pop()
            self._last = (None, 0.0)
            self.rev += 1

    # ---- output ---------------------------------------------------------
    def render(self, scale=1.0, override=None):
        """Render the current collage. `override` = (idx, dx, dy, panx, pany)
        previews an uncommitted drag without touching state."""
        self._need_plan()
        with self._render_lock:
            offs = dict(self.offsets)
            if override:
                idx = override[0]
                self._need_plan(idx)
                offs[idx] = self._clamp(*override)
            return collage.render_plan(self.plan, scale, offs)

    def jpeg(self, scale=1.0, override=None, quality=88):
        buf = io.BytesIO()
        self.render(scale, override).save(buf, "JPEG", quality=quality)
        return buf.getvalue()

    def export(self, quality):
        quality = max(1, min(100, int(quality)))
        img = self.render(1.0)
        path = collage.resolve_output_path(self.out_dir / "collage.jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, quality=quality)
        self.exports[path.name] = path
        return path

    # ---- views ----------------------------------------------------------
    def plan_view(self):
        if self.plan is None:
            return None
        plan = self.plan
        geo = collage._geometry(plan, self.offsets, 1.0)
        photo_of = {str(p["path"]): p["id"] for p in self.photos}   # placement -> photo id, even for equal filenames
        items = []
        for idx, (path, _, px0, py0, pw, ph) in enumerate(plan.placements):
            dx, dy, panx, pany = self.offsets.get(idx, (0, 0, 0, 0))
            (x0, x1), (y0, y1) = collage.pan_limits(plan, idx)
            item = dict(id=idx, name=Path(path).name, photo=photo_of.get(str(path)), x=px0 + dx, y=py0 + dy, w=pw, h=ph,
                        dx=dx, dy=dy, panx=panx, pany=pany,
                        pan_x=[x0, x1], pan_y=[y0, y1], pop=idx in plan.selected,
                        pop_box=self._pop_box(idx, geo[idx]) if idx in plan.selected else None)
            items.append(item)
        return dict(width=plan.width, height=plan.height, rev=self.rev, items=items,
                    order=plan.order, can_undo=bool(self.undo), can_redo=bool(self.redo),
                    seed=self.settings["seed"], settings=self.settings)

    def _pop_box(self, idx, g):
        """Canvas rect [x0, y0, x1, y1] of a popped subject (lifted), for hit-testing:
        a pop sits above neighbors' rects, so rect-only picking would miss it."""
        plan, cut = self.plan, self.plan.cut_of(idx)
        x, y, w, h, pan = g
        fnw, fnh, fl, ft = collage.cover_transform(plan.placements[idx][1], w, h, plan.keep_of(idx), pan)
        bx, by = x - fl, y - ft
        cx, cy = bx + cut.centroid[0] * fnw, by + cut.centroid[1] * fnh
        k = collage.POP_LIFT
        x0, y0 = bx + cut.bbox[0] * fnw, by + cut.bbox[1] * fnh
        x1, y1 = bx + cut.bbox[2] * fnw, by + cut.bbox[3] * fnh
        return [round(cx + (x0 - cx) * k), round(cy + (y0 - cy) * k),
                round(cx + (x1 - cx) * k), round(cy + (y1 - cy) * k)]

    def photo_view(self):
        return [dict({k: p[k] for k in ("id", "name", "w", "h")}, file=Path(p["path"]).name)
                for p in self.photos]
