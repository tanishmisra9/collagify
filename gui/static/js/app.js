// App shell: state, API, photo rail, selected-photo panel, shortcuts. The canvas
// lives in stage.js and the settings inspector in controls.js.
import { renderControls } from "./controls.js";
import { Stage } from "./stage.js";

const $ = (id) => document.getElementById(id);
const TOKEN = document.querySelector('meta[name="collagify-token"]').content;
const MAX_PHOTOS = 6;
const IS_MAC = /Mac|iPhone|iPad/.test(navigator.platform);

const state = {
  schema: [], photos: [], plan: null, settings: {}, generated: null,
  photosDirty: false, running: false, deep: false, nudged: 0,
};

// ---- api ---------------------------------------------------------------
async function api(method, path, body, headers = {}) {
  const res = await fetch(path, {
    method,
    headers: { "X-Token": TOKEN, ...(body && !(body instanceof Blob) ? { "Content-Type": "application/json" } : {}), ...headers },
    body: body === undefined ? undefined : body instanceof Blob ? body : JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

const ICONS = {
  ok: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg>',
  error: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.5"/></svg>',
};
const REMOVE_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"/></svg>';

function toast(msg, kind = "", link) {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.innerHTML = kind === "error" ? ICONS.error : ICONS.ok;
  const span = document.createElement("span");
  span.textContent = msg;
  t.append(span);
  if (link) { const a = document.createElement("a"); a.href = link.href; a.textContent = link.text; a.download = ""; t.append(a); }
  $("toasts").append(t);
  setTimeout(() => {
    t.classList.add("out");
    setTimeout(() => t.remove(), 220);
  }, kind === "error" ? 7000 : 4500);
}

// ---- stage ---------------------------------------------------------------
let commits = 0, chain = Promise.resolve(), liveVals = null;
const stage = new Stage(
  { viewport: $("viewport"), frame: $("frame"), shot: $("shot"), shot2: $("shot2"), overlay: $("overlay") },
  {
    onSelect: () => { renderSelected(); renderPhotos(); setHint(); },
    hint: (t) => ($("hint").textContent = t),
    drag: (on) => { document.body.dataset.drag = on ? "1" : ""; setHint(); },
    live: (v) => { liveVals = v; renderSelected(); },
    commit(id, v, merge) {
      commits++;
      chain = chain
        .then(() => api("POST", "/api/placement", { id, dx: v.dx, dy: v.dy, panx: v.panx, pany: v.pany, merge }))
        .then((r) => { if (--commits === 0) applyPlan(r.plan); })
        .catch((e) => { commits--; toast(e.message, "error"); refreshState(); });
    },
  });

const kbd = (t) => `<kbd>${t}</kbd>`;
function setHint() {
  const mod = IS_MAC ? "⌘" : "Ctrl+";
  const h = $("hint");
  if (!state.plan) h.innerHTML = "";
  else if (document.body.dataset.drag) h.innerHTML = `<span>Release to place</span><span>Hold ${kbd("Shift")} to pan the picture</span>`;
  else if (stage.selected == null) h.innerHTML = "<span>Click a photo to select it</span>";
  else h.innerHTML = `<span>Drag to move</span><span>${kbd("Shift")} drag pans the picture</span><span>${kbd("Alt")} click picks beneath</span>`
    + `<span>${kbd("←↑↓→")} nudge</span><span>${kbd(mod + "Z")} undo</span>`;
}

function applyPlan(plan) {
  state.plan = plan;
  liveVals = null;
  state.nudged = plan ? plan.items.filter((i) => i.dx || i.dy || i.panx || i.pany).length : 0;
  if (plan) { stage.setPlan(plan); state.generated = plan.settings; }
  else stage.clear();
  renderAll();
}

// ---- photos ---------------------------------------------------------------
// Plan items carry the id of their photo, so two photos with the same filename never get confused.
const nameOf = (it) => state.photos.find((p) => p.id === it.photo)?.name || it.name;
const idOfSelected = () => (stage.selected == null ? null : state.plan?.items[stage.selected]?.photo ?? null);

// Rows are keyed by photo id and updated in place, so selection, hover and enter
// animations survive the frequent re-renders (every drag commit calls this).
const rows = new Map();

function makeRow(p) {
  const li = document.createElement("li");
  li.className = "photo enter";
  li.dataset.id = p.id;
  li.innerHTML = '<button type="button" class="photo-main"><span class="thumb"><img alt="" draggable="false"><span class="idx num"></span></span>'
    + '<span class="meta"><span class="nm"></span><span class="sub num"></span></span></button>'
    + `<button type="button" class="btn icon sm rm">${REMOVE_ICON}</button>`;
  const img = li.querySelector("img");
  const ok = () => img.classList.add("ok");
  img.addEventListener("load", ok);
  img.src = `/api/photos/${p.id}/thumb.jpg`;
  if (img.complete) ok();
  li.querySelector(".nm").textContent = p.name;
  li.querySelector(".sub").textContent = `${p.w} × ${p.h}`;
  li.querySelector(".photo-main").setAttribute("aria-label", `Select ${p.name}`);
  const rm = li.querySelector(".rm");
  rm.setAttribute("aria-label", `Remove ${p.name}`);
  rm.addEventListener("click", (e) => { e.stopPropagation(); removePhoto(p); });
  li.addEventListener("animationend", () => li.classList.remove("enter"), { once: true });
  li.addEventListener("click", () => {
    const i = state.plan?.items.findIndex((it) => it.photo === p.id);
    if (i >= 0 && !state.photosDirty) stage.select(i);
  });
  return li;
}

function renderPhotos() {
  $("count").textContent = `${state.photos.length}/${MAX_PHOTOS}`;
  const list = $("photo-list");
  const selId = idOfSelected();
  const seen = new Set();
  let fresh = 0;
  state.photos.forEach((p, i) => {
    let li = rows.get(p.id);
    if (!li) {
      li = makeRow(p);
      li.style.setProperty("--d", `${fresh++ * 45}ms`);
      rows.set(p.id, li);
    }
    seen.add(p.id);
    if (list.children[i] !== li) list.insertBefore(li, list.children[i] || null);
    if (p.id === selId) li.setAttribute("aria-current", "true"); else li.removeAttribute("aria-current");
    li.querySelector(".idx").textContent = String(i + 1).padStart(2, "0");
  });
  for (const [id, li] of rows) if (!seen.has(id)) { rows.delete(id); li.remove(); }
  $("add").disabled = state.photos.length >= MAX_PHOTOS;
}

async function uploadFiles(files) {
  const imgs = [...files].filter((f) => /^image\//.test(f.type) || /\.(jpe?g|png|webp|bmp|tiff?|heic|heif)$/i.test(f.name));
  if (!imgs.length) return toast("Those aren't image files.", "error");
  for (const f of imgs) {
    try {
      const r = await api("POST", "/api/photos", f, { "X-Filename": encodeURIComponent(f.name), "X-Modified": String(f.lastModified), "Content-Type": "application/octet-stream" });
      state.photos = r.photos;
      state.photosDirty = true;
      renderAll();
    } catch (e) { toast(e.message, "error"); if (/at most/.test(e.message)) break; }
  }
}

async function removePhoto(p) {
  try {
    state.photos = (await api("DELETE", `/api/photos/${p.id}`)).photos;
    state.photosDirty = true;
    if (!state.photos.length) { applyPlan(null); state.photosDirty = false; }
    renderAll();
  } catch (e) { toast(e.message, "error"); }
}

// ---- generate -------------------------------------------------------------
let autoTimer = null;

function dirtyKeys() {
  if (!state.generated) return [];
  return state.schema.filter((s) => s.stage !== "output" && s.key !== "seed" || s.key === "seed" && state.settings.seed != null)
    .filter((s) => state.settings[s.key] !== state.generated[s.key]);
}

function onSetting(key, value) {
  state.settings[key] = value;
  clearTimeout(autoTimer);
  const dirty = dirtyKeys();
  const slow = dirty.some((s) => s.stage === "analysis");
  // Fast (layout-only) changes apply on their own, unless that would discard nudges.
  if (state.plan && dirty.length && !slow && !state.nudged && !state.photosDirty && !state.running) {
    autoTimer = setTimeout(generate, 450);
  }
  renderAll();
}

// The pipeline stages, matched against the server's status messages (collage.analyze).
const STEPS = [
  { label: "Match colors", match: /^Matching/, skip: (s) => s.color_strength <= 0 },
  { label: "Load cutout model", match: /^Loading cutout/, skip: (s) => !s.cutouts },
  { label: "Cut out subjects", match: /^Cutting out/, skip: (s) => !s.cutouts },
  { label: "Find faces", match: /^Finding faces/, skip: (s) => !s.faces },
  { label: "Lay out", match: /^(Laying out|Analysis done|Done)/, skip: () => false },
];
const ANALYSIS_MSG = /^(Matching|Loading cutout|Cutting out|Finding faces)/;

function buildSteps() {
  $("steps").replaceChildren(...STEPS.map((st) => {
    const li = document.createElement("li");
    const skip = st.skip(state.settings);
    if (skip) li.dataset.s = "skip";
    li.innerHTML = '<span class="lbl"></span><span class="n num"></span>';
    li.querySelector(".lbl").textContent = st.label;
    if (skip) li.querySelector(".n").textContent = "skipped";
    return li;
  }));
}

function paintSteps(msg) {
  const at = STEPS.findIndex((st) => st.match.test(msg));
  [...$("steps").children].forEach((li, i) => {
    if (li.dataset.s === "skip") return;
    li.dataset.s = at < 0 ? "" : i < at ? "done" : i === at ? "active" : "";
    li.querySelector(".n").textContent = i === at ? (msg.match(/\d+\/\d+/)?.[0] || "") : "";
  });
  $("busy-meter").classList.toggle("indet", /^Loading cutout/.test(msg));
}

async function generate() {
  if (state.running) return;
  if (!state.photos.length) return toast("Add at least one photo first.", "error");
  clearTimeout(autoTimer);
  const hadNudges = state.nudged, hadPlan = !!state.plan;
  const sent = { ...state.settings };      // what this run uses; later edits must survive its completion
  try {
    await api("POST", "/api/generate", state.settings);
  } catch (e) { return toast(e.message, "error"); }
  state.running = true;
  state.deep = false;
  buildSteps();
  $("busy-bar").style.transform = "scaleX(0)";
  const t0 = Date.now();
  renderAll();
  let wasDeep = false;
  for (;;) {
    await new Promise((r) => setTimeout(r, 250));
    const { status } = await api("GET", "/api/status").catch(() => ({ status: { state: "running" } }));
    const msg = status.message || "Working";
    // Escalate from the slim top line to the stepper card only once real analysis work shows up.
    if (!state.deep && ANALYSIS_MSG.test(msg)) { state.deep = wasDeep = true; renderAll(); }
    $("busy-msg").textContent = msg;
    $("busy-time").textContent = `${Math.floor((Date.now() - t0) / 1000)}s`;
    $("busy-bar").style.transform = `scaleX(${status.progress || 0})`;
    paintSteps(msg);
    if (status.state === "running") continue;
    state.running = false;
    state.deep = false;
    if (status.state === "error") toast(status.error, "error");
    else {
      state.photosDirty = false;
      await refreshState(sent);
      if (status.warning) toast(status.warning, "error");
      if (wasDeep || !hadPlan) stage.reveal();
      if (hadNudges) toast("New layout generated. Previous nudges were cleared.");
    }
    break;
  }
  renderAll();
}

async function shuffle() {
  state.settings.seed = Math.floor(Math.random() * 999999) + 1;
  controls.sync(state.settings);
  generate();
}

// ---- selected photo panel ---------------------------------------------
function renderSelected() {
  const it = state.plan && stage.selected != null ? state.plan.items[stage.selected] : null;
  $("selected").hidden = !it;
  if (!it) return;
  const c = stage.cur(it.id);
  $("sel-name").textContent = nameOf(it) + (it.pop ? "  ·  popped" : "");
  for (const [id, k] of [["n-dx", "dx"], ["n-dy", "dy"], ["n-px", "panx"], ["n-py", "pany"]]) {
    const input = $(id);
    if (document.activeElement !== input) input.value = c[k];
  }
  $("n-px").disabled = it.pan_x[0] === it.pan_x[1];
  $("n-py").disabled = it.pan_y[0] === it.pan_y[1];
  $("sel-note").textContent =
    `Move shifts the frame. Pan slides the picture inside it (X ${it.pan_x[0]} to ${it.pan_x[1]}, Y ${it.pan_y[0]} to ${it.pan_y[1]}).`;
  $("reset-one").disabled = !(it.dx || it.dy || it.panx || it.pany);
}

function commitInputs() {
  const it = state.plan.items[stage.selected];
  const v = { dx: +$("n-dx").value || 0, dy: +$("n-dy").value || 0, panx: +$("n-px").value || 0, pany: +$("n-py").value || 0 };
  stage.setLive(it.id, v, false);
  stage.cb.commit(it.id, stage.live, false);
}

// ---- chrome ---------------------------------------------------------------
function renderAll() {
  renderPhotos();
  renderSelected();
  const has = !!state.plan, busy = state.running, n = state.photos.length;
  const dirty = dirtyKeys();
  const stale = has && (state.photosDirty || dirty.length > 0);

  $("empty").hidden = has || busy;
  $("frame").classList.toggle("dim", state.deep || stale && !!dirty.some((s) => s.stage === "analysis"));
  $("busy").hidden = !(busy && state.deep);
  $("topline").classList.toggle("on", busy);
  $("generate").setAttribute("aria-busy", busy);
  $("generate").disabled = busy || !n;
  $("shuffle").disabled = busy || !n;
  $("generate").classList.toggle("attn", !busy && n > 0 && (!has || stale));
  $("export").disabled = !has || busy;
  $("undo").disabled = !has || !state.plan.can_undo;
  $("redo").disabled = !has || !state.plan.can_redo;
  $("reset-all").disabled = !has || !state.nudged;

  if (!has) {
    $("empty-title").textContent = n ? `${n} photo${n > 1 ? "s" : ""} ready` : "Drop 1–6 photos";
    $("empty-body").textContent = n ? "Press Generate to build the collage. The first run downloads the cutout model, which can take a minute."
      : "People photos work best. Faces stay in frame and a standout can pop off the board.";
    $("empty-add").textContent = n ? "Generate" : "Choose photos";
    $("empty-add").dataset.action = n ? "generate" : "add";
  }

  const notice = $("notice");
  const auto = stale && !state.nudged && !state.photosDirty && !dirty.some((s) => s.stage === "analysis") && !busy;
  const show = stale && !busy && !auto;
  if (show) {
    const msg = state.photosDirty ? "Photos changed." : dirty.some((s) => s.stage === "analysis")
      ? "Settings changed. Applying re-runs the AI steps." : `Settings changed. Applying clears ${state.nudged} nudged photo${state.nudged > 1 ? "s" : ""}.`;
    if (notice.hidden || notice.firstChild?.textContent !== msg) {   // rebuild only on change so the pop-in does not replay
      notice.replaceChildren();
      const span = document.createElement("span");
      span.textContent = msg;
      const b = document.createElement("button");
      b.className = "btn primary"; b.type = "button"; b.textContent = "Apply";
      b.addEventListener("click", generate);
      notice.append(span, b);
    }
  }
  notice.hidden = !show;
  setHint();
}

async function refreshState(sent) {
  const s = await api("GET", "/api/state");
  state.photos = s.photos;
  // Take the server's values, except settings the user edited while a run was in flight.
  for (const [k, v] of Object.entries(s.settings)) if (!sent || state.settings[k] === sent[k]) state.settings[k] = v;
  controls.sync(state.settings);
  applyPlan(s.plan);
}

function act(name) {
  const fn = { undo: () => api("POST", "/api/undo", {}), redo: () => api("POST", "/api/redo", {}), reset: () => api("POST", "/api/reset", {}) }[name];
  chain = chain.then(fn).then((r) => applyPlan(r.plan)).catch((e) => toast(e.message, "error"));
}

async function exportCollage() {
  const b = $("export");
  b.disabled = true;
  b.setAttribute("aria-busy", "true");
  try {
    const r = await api("POST", "/api/export", { quality: state.settings.quality });
    toast(`Saved ${r.path.split("/").slice(-2).join("/")}`, "", { href: r.url, text: "Download" });
    b.classList.add("done");
    $("export-label").textContent = "Saved";
    setTimeout(() => { b.classList.remove("done"); $("export-label").textContent = "Export"; }, 1400);
  } catch (e) { toast(e.message, "error"); }
  b.removeAttribute("aria-busy");
  renderAll();
}

// ---- scroll edge fades ----------------------------------------------------
// Flags drive the sticky gradient pseudo-elements in app.css; both flags set = nothing to scroll.
function watchEdges(el) {
  const update = () => {
    el.toggleAttribute("data-at-top", el.scrollTop <= 1);
    el.toggleAttribute("data-at-bottom", el.scrollTop + el.clientHeight >= el.scrollHeight - 1);
  };
  el.addEventListener("scroll", update, { passive: true });
  const ro = new ResizeObserver(update);
  ro.observe(el);
  const watchKids = () => { for (const c of el.children) ro.observe(c); update(); };
  new MutationObserver(watchKids).observe(el, { childList: true });
  watchKids();
}

// ---- wiring ---------------------------------------------------------------
let controls;
function theme() {
  const root = document.documentElement;
  root.dataset.theme = root.dataset.theme === "light" ? "dark" : "light";
  try { localStorage.setItem("collagify-theme", root.dataset.theme); } catch { /* private mode */ }
  stage.draw();
}

async function init() {
  const sch = await api("GET", "/api/schema");
  state.schema = sch.settings;
  state.settings = Object.fromEntries(sch.settings.map((s) => [s.key, s.default]));
  controls = renderControls($("settings"), sch.settings, state.settings, onSetting);
  controls.sync(state.settings);
  document.querySelectorAll(".scroller").forEach(watchEdges);
  await refreshState();

  $("generate").addEventListener("click", generate);
  $("shuffle").addEventListener("click", shuffle);
  $("undo").addEventListener("click", () => act("undo"));
  $("redo").addEventListener("click", () => act("redo"));
  $("reset-all").addEventListener("click", () => act("reset"));
  $("export").addEventListener("click", exportCollage);
  $("theme").addEventListener("click", theme);
  $("deselect").addEventListener("click", () => stage.select(null));
  $("reset-one").addEventListener("click", () => {
    chain = chain.then(() => api("POST", "/api/reset", { id: stage.selected })).then((r) => applyPlan(r.plan)).catch((e) => toast(e.message, "error"));
  });
  for (const id of ["n-dx", "n-dy", "n-px", "n-py"]) $(id).addEventListener("change", commitInputs);

  const file = $("file");
  const openPicker = () => file.click();
  $("add").addEventListener("click", openPicker);
  $("empty-add").addEventListener("click", (e) => (e.currentTarget.dataset.action === "generate" ? generate() : openPicker()));
  file.addEventListener("change", () => { uploadFiles(file.files); file.value = ""; });
  $("folder-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      state.photos = (await api("POST", "/api/photos/folder", { path: $("folder-path").value.trim() })).photos;
      state.photosDirty = true; renderAll();
    } catch (err) { toast(err.message, "error"); }
  });

  const dropTarget = $("stage");
  for (const t of [dropTarget, document.querySelector(".rail")]) {
    t.addEventListener("dragover", (e) => { if (e.dataTransfer?.types.includes("Files")) { e.preventDefault(); dropTarget.classList.add("dragover"); } });
    t.addEventListener("dragleave", (e) => { if (!t.contains(e.relatedTarget)) dropTarget.classList.remove("dragover"); });
    t.addEventListener("drop", (e) => { e.preventDefault(); dropTarget.classList.remove("dragover"); uploadFiles(e.dataTransfer.files); });
  }

  addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea") && e.target.type !== "checkbox") return;
    const mod = IS_MAC ? e.metaKey : e.ctrlKey;
    if (mod && e.key.toLowerCase() === "z") { e.preventDefault(); act(e.shiftKey ? "redo" : "undo"); }
    else if (mod && e.key.toLowerCase() === "y") { e.preventDefault(); act("redo"); }
    else if (mod || e.altKey) return;
    else if (e.target === $("overlay") && e.key.startsWith("Arrow")) return;   // stage handles these
    else if (e.key === "g") generate();
    else if (e.key === "s") shuffle();
    else if (e.key === "e" && state.plan) exportCollage();
    else if ((e.key === "[" || e.key === "]") && state.plan) {
      const n = state.plan.items.length, d = e.key === "]" ? 1 : -1;
      stage.select(stage.selected == null ? 0 : (stage.selected + d + n) % n);
    } else if (e.key === "Escape") stage.select(null);
  });
}

init().catch((e) => toast(`Could not start: ${e.message}`, "error"));
