// The collage canvas: fit-to-view image, selection overlay, hit-testing and drag.
// Dragging draws the outline instantly and asks the server for low-res previews
// (stateless: nothing is committed until release). Release commits one undo step.

const PREVIEW_SCALE = 0.45;
const DRAG_THRESHOLD = 3;     // client px before a press becomes a drag
const MIN_VISIBLE = 0.3;      // keep in sync with gui/session.py

const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const fmt = (n) => (n > 0 ? "+" : n < 0 ? "−" : "") + Math.abs(n);

export class Stage {
  constructor(els, cb) {
    Object.assign(this, els);           // viewport, frame, shot, shot2, overlay
    this.cb = cb;                       // onSelect(id|null), commit(id, vals, merge), hint(text), drag(bool), live(vals)
    this.plan = null;
    this.selected = null;
    this.hover = null;                  // id under the pointer (target)
    this.hoverId = null;                // id currently drawn (lingers while it fades out)
    this.hoverA = 0;                    // eased hover alpha, 0..1
    this.swapN = 0;
    this.live = null;                   // {id, dx, dy, panx, pany} during a drag / nudge
    this.drag = null;
    this.epoch = 0;                     // bumps when a preview must be discarded
    this.ds = 1;                        // display px per canvas px
    this.ctx = this.overlay.getContext("2d");
    new ResizeObserver(() => this.fit()).observe(this.viewport);
    const o = this.overlay;
    o.addEventListener("pointerdown", (e) => this.down(e));
    o.addEventListener("pointermove", (e) => this.move(e));
    o.addEventListener("pointerup", (e) => this.up(e));
    o.addEventListener("pointercancel", () => this.cancel());
    o.addEventListener("pointerleave", () => this.setHover(null));
    o.addEventListener("keydown", (e) => this.key(e));
  }

  // ---- data -------------------------------------------------------------
  setPlan(plan) {
    const sizeChanged = !this.plan || this.plan.width !== plan.width || this.plan.height !== plan.height;
    this.plan = plan;
    this.live = null;
    if (this.selected != null && !plan.items[this.selected]) this.selected = null;
    this.frame.hidden = false;
    if (sizeChanged) this.fit();
    this.loadFull(plan.rev);
    this.draw();
  }

  clear() {
    this.plan = null; this.selected = null; this.live = null;
    this.frame.hidden = true;
  }

  select(id) {
    this.selected = id;
    this.draw();
    this.cb.onSelect(id);
  }

  item(id) { return this.plan?.items[id]; }

  // Current offsets of an item, including an in-flight drag.
  cur(id) {
    const it = this.item(id);
    if (this.live && this.live.id === id) return this.live;
    return { id, dx: it.dx, dy: it.dy, panx: it.panx, pany: it.pany };
  }

  rect(it) {
    const c = this.cur(it.id);
    return { x: it.x - it.dx + c.dx, y: it.y - it.dy + c.dy, w: it.w, h: it.h };
  }

  // ---- layout / image ---------------------------------------------------
  fit() {
    if (!this.plan) return;
    const cs = getComputedStyle(this.viewport);
    const aw = this.viewport.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    const ah = this.viewport.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
    this.ds = Math.max(0.05, Math.min(aw / this.plan.width, ah / this.plan.height));
    const w = Math.floor(this.plan.width * this.ds), h = Math.floor(this.plan.height * this.ds);
    this.frame.style.width = `${w}px`;
    this.frame.style.height = `${h}px`;
    const dpr = window.devicePixelRatio || 1;
    this.overlay.width = Math.round(w * dpr);
    this.overlay.height = Math.round(h * dpr);
    this.dpr = dpr;
    this.draw();
  }

  loadFull(rev) {
    const token = ++this.epoch;
    const url = `/api/preview.jpg?scale=1&v=${rev}`;
    const im = new Image();
    // A wait only reads as a wait if it lasts; show the placeholder pulse after 150ms.
    clearTimeout(this.pulseT);
    this.pulseT = setTimeout(() => this.frame.classList.add("pulse"), 150);
    const done = () => { clearTimeout(this.pulseT); this.frame.classList.remove("pulse"); };
    im.onload = () => { done(); if (token === this.epoch) this.swap(url); };
    im.onerror = done;
    im.src = url;
  }

  // Direct set: used for drag previews, where a fade would only add lag.
  setSrc(url, blob = false) {
    if (this.blobUrl) URL.revokeObjectURL(this.blobUrl);
    this.blobUrl = blob ? url : null;
    this.shot.src = url;
  }

  // Settled render: cross-fade old -> new through the second layer.
  swap(url) {
    const n = ++this.swapN;
    if (REDUCED || !this.shot.getAttribute("src")) { this.shot2.classList.remove("in"); return this.setSrc(url); }
    this.shot2.src = url;
    this.shot2.decode().catch(() => {}).then(() => {
      if (n !== this.swapN) return;
      this.shot2.classList.add("in");
      setTimeout(() => {
        if (n !== this.swapN) return;
        this.setSrc(url);
        this.shot2.classList.remove("in");
      }, 210);
    });
  }

  // Entrance for a freshly generated collage (not for nudges, undo or reset).
  reveal() {
    if (REDUCED || !this.frame.animate) return;
    this.frame.animate(
      [{ opacity: 0.5, transform: "scale(.985)", filter: "blur(4px)" }, { opacity: 1, transform: "none", filter: "none" }],
      { duration: 420, easing: "cubic-bezier(.16,1,.3,1)" });
  }

  // ---- hit-testing ------------------------------------------------------
  toCanvas(e) {
    const r = this.overlay.getBoundingClientRect();
    return [(e.clientX - r.left) / this.ds, (e.clientY - r.top) / this.ds];
  }

  hits(px, py) {
    const out = [];
    const items = this.plan.items;
    for (const it of items) {          // lifted subjects sit above every frame
      if (!it.pop_box) continue;
      const c = this.cur(it.id), dx = c.dx - it.dx, dy = c.dy - it.dy;
      const [x0, y0, x1, y1] = it.pop_box;
      if (px >= x0 + dx && px <= x1 + dx && py >= y0 + dy && py <= y1 + dy) out.push(it.id);
    }
    for (const id of [...this.plan.order].reverse()) {
      const r = this.rect(items[id]);
      if (px >= r.x && px < r.x + r.w && py >= r.y && py < r.y + r.h && !out.includes(id)) out.push(id);
    }
    return out;
  }

  // ---- pointer ----------------------------------------------------------
  down(e) {
    if (!this.plan || e.button !== 0) return;
    const [px, py] = this.toCanvas(e);
    const hits = this.hits(px, py);
    let id = hits[0] ?? null;
    if (e.altKey && hits.length > 1) {      // Alt-click cycles to the photo underneath
      const i = hits.indexOf(this.selected);
      id = hits[(i + 1) % hits.length];
    }
    if (id !== this.selected) this.select(id);
    if (id == null) return;
    this.overlay.setPointerCapture(e.pointerId);
    const it = this.item(id);
    this.drag = { id, sx: e.clientX, sy: e.clientY, base: { ...this.cur(id) }, pan: e.shiftKey, active: false };
    if (this.drag.pan && it.pan_x[1] === it.pan_x[0] && it.pan_y[1] === it.pan_y[0]) {
      this.cb.hint("This photo has no room to pan: its picture already fills the frame exactly.");
    }
  }

  move(e) {
    if (!this.plan) return;
    const d = this.drag;
    if (!d) {
      const [px, py] = this.toCanvas(e);
      const h = this.hits(px, py)[0] ?? null;
      this.setHover(h);
      this.overlay.classList.toggle("can-grab", h != null);
      return;
    }
    const mx = e.clientX - d.sx, my = e.clientY - d.sy;
    if (!d.active && Math.hypot(mx, my) < DRAG_THRESHOLD) return;
    if (!d.active) this.cb.drag?.(true);
    d.active = true;
    d.pan = e.shiftKey;
    this.overlay.classList.toggle("moving", !d.pan);
    this.overlay.classList.toggle("panning", d.pan);
    const ddx = mx / this.ds, ddy = my / this.ds;
    const b = d.base;
    this.setLive(d.id, d.pan
      ? { ...b, panx: b.panx - ddx, pany: b.pany - ddy }       // drag the picture, not the window
      : { ...b, dx: b.dx + ddx, dy: b.dy + ddy });
  }

  up(e) {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    this.overlay.classList.remove("moving", "panning");
    if (d.active) this.cb.drag?.(false);
    if (d.active && this.live) {
      const v = this.live;
      this.epoch++;                          // drop any preview still in flight
      this.cb.commit(d.id, v, false);
    }
    this.draw();
  }

  cancel() {
    if (!this.drag) return;
    if (this.drag.active) this.cb.drag?.(false);
    this.drag = null; this.live = null; this.epoch++;
    this.overlay.classList.remove("moving", "panning");
    if (this.plan) this.loadFull(this.plan.rev);
    this.draw();
  }

  // ---- keyboard ---------------------------------------------------------
  key(e) {
    if (!this.plan || e.metaKey || e.ctrlKey) return;
    const arrows = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (e.key in arrows && this.selected != null) {
      e.preventDefault();
      const step = e.shiftKey ? 10 : 1, [ax, ay] = arrows[e.key];
      const c = this.cur(this.selected);
      this.setLive(this.selected, e.altKey
        ? { ...c, panx: c.panx - ax * step, pany: c.pany - ay * step }   // Alt+arrows pan the picture
        : { ...c, dx: c.dx + ax * step, dy: c.dy + ay * step }, false);
      this.cb.commit(this.selected, this.live, true);
    } else if (e.key === "Escape") {
      this.select(null);
    }
  }

  // ---- live offsets + previews -----------------------------------------
  setLive(id, v, preview = true) {
    const it = this.item(id);
    const px0 = it.x - it.dx, py0 = it.y - it.dy;
    const W = this.plan.width, H = this.plan.height;
    this.live = {
      id,
      dx: Math.round(clamp(v.dx, -px0 - (1 - MIN_VISIBLE) * it.w, W - MIN_VISIBLE * it.w - px0)),
      dy: Math.round(clamp(v.dy, -py0 - (1 - MIN_VISIBLE) * it.h, H - MIN_VISIBLE * it.h - py0)),
      panx: Math.round(clamp(v.panx, it.pan_x[0], it.pan_x[1])),
      pany: Math.round(clamp(v.pany, it.pan_y[0], it.pan_y[1])),
    };
    this.draw();
    this.cb.live(this.live);
    if (preview) this.requestPreview();
  }

  requestPreview() {
    this.wanted = { ...this.live };
    if (this.inflight) return;
    const run = () => {
      const p = this.wanted;
      if (!p) { this.inflight = false; return; }
      this.wanted = null;
      this.inflight = true;
      const token = this.epoch;
      const q = new URLSearchParams({ scale: PREVIEW_SCALE, id: p.id, dx: p.dx, dy: p.dy, panx: p.panx, pany: p.pany });
      fetch(`/api/preview.jpg?${q}`).then((r) => (r.ok ? r.blob() : Promise.reject()))
        .then((b) => { if (token === this.epoch) this.setSrc(URL.createObjectURL(b), true); })
        .catch(() => {})
        .finally(run);
    };
    run();
  }

  // ---- overlay ----------------------------------------------------------
  draw() {
    const c = this.ctx;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, this.overlay.width, this.overlay.height);
    if (!this.plan) return;
    const k = this.ds * (this.dpr || 1);
    const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#c2410c";
    c.setTransform(k, 0, 0, k, 0, 0);
    const line = 2 / k * (this.dpr || 1);
    const hv = this.hoverId != null && this.hoverId !== this.selected ? this.item(this.hoverId) : null;
    if (hv && this.hoverA > 0.01) {
      const r = this.rect(hv);
      c.globalAlpha = 0.6 * this.hoverA; c.strokeStyle = accent; c.lineWidth = line * 0.75; c.setLineDash([8 / this.ds, 6 / this.ds]);
      c.strokeRect(r.x, r.y, r.w, r.h);
    }
    const sel = this.selected != null ? this.item(this.selected) : null;
    if (sel) {
      const r = this.rect(sel);
      c.globalAlpha = 0.1; c.fillStyle = accent; c.fillRect(r.x, r.y, r.w, r.h);
      c.globalAlpha = 1; c.setLineDash([]); c.lineWidth = line * 2;
      c.strokeStyle = "rgba(255,255,255,.85)"; c.strokeRect(r.x, r.y, r.w, r.h);
      c.lineWidth = line * 1.4; c.strokeStyle = accent; c.strokeRect(r.x, r.y, r.w, r.h);
      if (this.drag?.active) this.drawHud(sel, r);
    }
  }

  // Ease the hover outline in and out instead of snapping.
  setHover(h) {
    if (h === this.hover) return;
    this.hover = h;
    if (h != null) this.hoverId = h;
    if (this.raf) return;
    const step = () => {
      const target = this.hover != null ? 1 : 0;
      this.hoverA = REDUCED ? target : this.hoverA + (target - this.hoverA) * 0.3;
      if (Math.abs(target - this.hoverA) < 0.02) this.hoverA = target;
      this.draw();
      this.raf = this.hoverA === target ? 0 : requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }

  // Offset readout (and a crosshair in pan mode) drawn in screen space so text stays crisp.
  drawHud(sel, r) {
    const c = this.ctx, d = this.dpr || 1, ds = this.ds;
    const v = this.cur(sel.id), pan = this.drag.pan;
    const text = pan ? `pan  x ${fmt(v.panx)}  y ${fmt(v.pany)}` : `dx ${fmt(v.dx)}  dy ${fmt(v.dy)}`;
    c.setTransform(d, 0, 0, d, 0, 0);
    c.setLineDash([]); c.globalAlpha = 1;
    if (pan) {
      const cx = (r.x + r.w / 2) * ds, cy = (r.y + r.h / 2) * ds;
      for (const [w, col] of [[4, "rgba(0,0,0,.45)"], [1.5, "#fff"]]) {
        c.lineWidth = w; c.strokeStyle = col; c.beginPath();
        c.moveTo(cx - 12, cy); c.lineTo(cx + 12, cy); c.moveTo(cx, cy - 12); c.lineTo(cx, cy + 12); c.stroke();
      }
    }
    c.font = '500 11px "Space Mono", ui-monospace, monospace';
    const w = c.measureText(text).width + 16, h = 22;
    const x = clamp(r.x * ds + 8, 4, this.overlay.width / d - w - 4);
    const y = clamp(r.y * ds + 8, 4, this.overlay.height / d - h - 4);
    c.fillStyle = "rgba(10,10,12,.85)";
    c.beginPath(); c.roundRect(x, y, w, h, 11); c.fill();
    c.fillStyle = "#fff"; c.textBaseline = "middle"; c.fillText(text, x + 8, y + h / 2 + 0.5);
  }
}
