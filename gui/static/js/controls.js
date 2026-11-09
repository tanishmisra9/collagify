// Renders the settings inspector from the server's schema (/api/schema).
// One entry in gui/schema.py == one control here; nothing is hand-wired per setting.

const GROUP_ORDER = ["Canvas", "Layout", "Look", "Subjects", "Advanced", "Output"];
const OPEN_BY_DEFAULT = new Set(["Canvas", "Layout", "Look", "Subjects"]);
const STORE = "collagify-groups";

const decimals = (step) => (String(step).split(".")[1] || "").length;
const loadOpen = () => { try { return JSON.parse(localStorage.getItem(STORE)) || {}; } catch { return {}; } };
const saveOpen = (o) => { try { localStorage.setItem(STORE, JSON.stringify(o)); } catch { /* private mode */ } };

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (v !== undefined && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  e.append(...kids);
  return e;
}

export function renderControls(root, specs, settings, onChange) {
  const refs = {};
  const groups = new Map();
  for (const spec of specs) {
    if (!groups.has(spec.group)) groups.set(spec.group, []);
    groups.get(spec.group).push(spec);
  }
  const saved = loadOpen();
  let n = 0;                                   // stagger index across the whole inspector
  root.replaceChildren();
  for (const name of GROUP_ORDER.filter((g) => groups.has(g))) {
    const body = el("div", { class: "group-body" });
    for (const spec of groups.get(name)) {
      const c = control(spec);
      c.style.setProperty("--i", Math.min(n++, 10));
      body.append(c);
    }
    const rid = `grp-${name}`;
    const head = el("button", { class: "group-head", type: "button", "aria-controls": rid },
      el("span", { class: "kicker" }, name), el("span", { class: "chev", "aria-hidden": "true" }));
    const region = el("div", { class: "group-region", id: rid }, el("div", { class: "group-inner" }, body));
    const setOpen = (open) => { head.setAttribute("aria-expanded", String(open)); region.inert = !open; };
    setOpen(saved[name] ?? OPEN_BY_DEFAULT.has(name));
    head.addEventListener("click", () => {
      const open = head.getAttribute("aria-expanded") !== "true";
      setOpen(open);
      saved[name] = open;
      saveOpen(saved);
    });
    root.append(el("section", { class: "group" }, head, region));
  }

  function control(spec) {
    const id = `ctl-${spec.key}`;
    const slow = spec.stage === "analysis" ? el("span", { class: "slow", title: "Re-runs the AI steps" }, "Slower") : "";
    const help = el("p", { class: "help", id: `${id}-h` }, spec.help || "");
    let input, out;
    if (spec.type === "toggle") {
      input = el("input", { type: "checkbox", role: "switch", id, "aria-describedby": help.id });
      input.addEventListener("change", () => onChange(spec.key, input.checked));
      refs[spec.key] = { set: (v) => (input.checked = !!v) };
      return el("div", { class: "ctl" },
        el("div", { class: "switch" }, el("label", { for: id }, spec.label, slow), input), help);
    }
    if (spec.type === "select") {
      input = el("select", { id, "aria-describedby": help.id },
        ...spec.options.map((o) => el("option", { value: o }, o)));
      input.addEventListener("change", () => onChange(spec.key, input.value));
      refs[spec.key] = { set: (v) => (input.value = v) };
    } else if (spec.type === "number") {
      input = el("input", { type: "number", id, min: spec.min, max: spec.max, step: 1, placeholder: "random",
        inputmode: "numeric", "aria-describedby": help.id });
      input.addEventListener("change", () => onChange(spec.key, input.value === "" ? null : Number(input.value)));
      refs[spec.key] = { set: (v) => (input.value = v ?? "") };
    } else {
      const d = decimals(spec.step);
      input = el("input", { type: "range", id, min: spec.min, max: spec.max, step: spec.step, "aria-describedby": help.id });
      out = el("output", { for: id, class: "num" });
      const paint = () => {                    // filled part of the track
        out.textContent = Number(input.value).toFixed(d);
        input.style.setProperty("--p", `${((input.value - spec.min) / (spec.max - spec.min)) * 100}%`);
      };
      input.addEventListener("input", paint);
      input.addEventListener("change", () => onChange(spec.key, Number(input.value)));
      refs[spec.key] = { set: (v) => { input.value = v; paint(); } };
    }
    return el("div", { class: "ctl" },
      el("div", { class: "ctl-row" }, el("label", { for: id }, spec.label, slow), out || ""), input, help);
  }

  return {
    sync(values) {
      for (const [k, r] of Object.entries(refs)) if (k in values) r.set(values[k]);
    },
  };
}
