// The dashboard page: tables from /api/*, and the Rerun web viewer for the open sequence.
import { WebViewer } from "rerun-web-viewer";

const $ = (id) => document.getElementById(id);
const GENDER = { f: "female", m: "male" };
const ACTOR_COLORS = ["#ff7043", "#42a5f5", "#66bb6a", "#ab47bc"];
const state = { subset: "all", rows: [], subsets: [], byName: new Map(), open: null, viewer: null };
window.__dash = state;  // for automated tests

// ------------------------------------------------------------------ tables
// columns: [{key, label, num, fmt}]. A header click sorts; a second click reverses.
function renderTable(el, columns, rows, { onClick, sort, muted } = {}) {
  el._sort = el._sort || sort || null;
  const s = el._sort;
  const sorted = s ? [...rows].sort((a, b) => cmp(a[s.key], b[s.key]) * s.dir) : rows;
  const head = columns.map((c) => {
    const mark = s && s.key === c.key ? (s.dir > 0 ? " ▲" : " ▼") : "";
    return `<th class="${c.num ? "num" : ""}" data-key="${c.key}" aria-sort="${mark ? (s.dir > 0 ? "ascending" : "descending") : "none"}">${c.label}${mark}</th>`;
  }).join("");
  const body = sorted.map((r) => `<tr data-name="${r.name ?? ""}" class="${muted && muted(r) ? "muted" : ""}${r.name && r.name === state.open ? " open" : ""}">` +
    columns.map((c) => `<td class="${c.num ? "num" : ""}">${c.fmt ? c.fmt(r[c.key], r) : r[c.key] ?? ""}</td>`).join("") + "</tr>").join("");
  el.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;
  el.querySelectorAll("th").forEach((th) => th.onclick = () => {
    const key = th.dataset.key;
    el._sort = { key, dir: el._sort && el._sort.key === key ? -el._sort.dir : 1 };
    renderTable(el, columns, rows, { onClick, muted });
  });
  if (onClick) el.querySelectorAll("tbody tr").forEach((tr) => tr.onclick = () => onClick(tr.dataset.name));
}
function cmp(a, b) {
  if (a === b) return 0;
  if (a === null || a === undefined || a === "") return 1;
  if (b === null || b === undefined || b === "") return -1;
  return a < b ? -1 : 1;
}
const fixed = (d) => (v) => (v === null || v === undefined ? "" : Number(v).toFixed(d));
const actorsHtml = (g) => [...g].map((x, i) => `<span class="actor"><i class="swatch" style="background:${ACTOR_COLORS[i % 4]}"></i>${GENDER[x] || x}</span>`).join(" ");

// ------------------------------------------------------------------ Sequences tab
const SEQ_COLUMNS = [
  { key: "name", label: "Sequence" },
  { key: "split", label: "Split" },
  { key: "frames", label: "Frames", num: true },
  { key: "seconds", label: "Duration (s)", num: true, fmt: fixed(1) },
  { key: "genders", label: "Actors", fmt: actorsHtml },
  { key: "valid", label: "Valid %", num: true, fmt: fixed(1) },
  { key: "tags", label: "Subsets" },
  { key: "pc", label: "PC range" },
];
function inSubset(r, key) { return key === "all" || r.subsets.includes(key); }
function filteredRows() {
  const q = $("search").value.trim().toLowerCase(), n = $("actors").value;
  return state.rows.filter((r) => (!q || r.name.toLowerCase().includes(q)) && (!n || r.genders.length === +n));
}
function renderSequences() {
  const rows = filteredRows();
  $("subsets").innerHTML = state.subsets.map(([key, label]) =>
    `<button role="radio" aria-checked="${key === state.subset}" data-key="${key}">${label} <span>${rows.filter((r) => inSubset(r, key)).length}</span></button>`).join("");
  $("subsets").querySelectorAll("button").forEach((b) => b.onclick = () => { state.subset = b.dataset.key; renderSequences(); });
  const shown = rows.filter((r) => inSubset(r, state.subset));
  const note = state.subsets.find(([k]) => k === state.subset)[2];
  $("subset-note").textContent = shown.length ? note : "No sequence matches the filters.";
  renderTable($("seq-table"), SEQ_COLUMNS, shown, { onClick: (name) => openSequence(name) });
}

// ------------------------------------------------------------------ SAM 3D Body tab
function renderSam3d(s) {
  $("sam3d-missing").hidden = !!s; $("sam3d-body").hidden = !s;
  if (!s) return;
  const LABEL = { all: "All", occ: "3DPW-OCC", train: "Train", validation: "Validation", test: "Test" };
  const totals = Object.keys(LABEL).filter((k) => s.totals[k]).map((k) => ({ split: LABEL[k], ...s.totals[k] }));
  renderTable($("sam3d-totals"), [
    { key: "split", label: "Split" }, { key: "sequences", label: "Sequences", num: true },
    { key: "scored", label: "Actor-frames", num: true, fmt: (v) => v.toLocaleString("en-US") },
    { key: "mpjpe", label: "MPJPE (mm)", num: true, fmt: fixed(1) },
    { key: "pa_mpjpe", label: "PA-MPJPE (mm)", num: true, fmt: fixed(1) },
  ], totals);
  const err = Object.fromEntries(s.joints.map((j, i) => [j, s.totals.all.per_joint_mpjpe[i]]));
  const joints = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"].map((j) =>
    ({ joint: j[0].toUpperCase() + j.slice(1), left: err[`L ${j}`], right: err[`R ${j}`] }));
  renderTable($("sam3d-joints"), [
    { key: "joint", label: "Joint (all)" }, { key: "left", label: "Left (mm)", num: true, fmt: fixed(1) },
    { key: "right", label: "Right (mm)", num: true, fmt: fixed(1) },
  ], joints);
  $("sam3d-caution").textContent = "Caution: " + s.caution;
  const rows = s.sequences.map((r) => ({ ...r, scoredText: `${r.scored}/${r.valid_gt}`, note: r.excluded ? "excluded: " + shortReason(r.excluded) : "" }));
  rows.sort((a, b) => (!!a.excluded - !!b.excluded) || cmp(b.mpjpe, a.mpjpe));
  renderTable($("sam3d-table"), [
    { key: "name", label: "Sequence" }, { key: "split", label: "Split" },
    { key: "actors", label: "Actors", num: true }, { key: "scoredText", label: "Scored", num: true },
    { key: "mpjpe", label: "MPJPE (mm)", num: true, fmt: fixed(1) },
    { key: "pa_mpjpe", label: "PA-MPJPE (mm)", num: true, fmt: fixed(1) },
    { key: "root", label: "Root error (mm)", num: true, fmt: fixed(0) },
    { key: "note", label: "Note" },
  ], rows, { onClick: (name) => openSequence(name, state.sam3dWorst[name]), muted: (r) => !!r.excluded });
  const excluded = s.sequences.filter((r) => r.excluded).map((r) => `${r.name}: ${r.excluded}`);
  $("sam3d-notes").textContent = s.protocol + (excluded.length ? " Excluded from the averages: " + excluded.join("; ") + "." : "");
}
function shortReason(reason) { const m = reason.match(/\(([^()]+)\)\s*$/); return m ? m[1] : reason; }

// ------------------------------------------------------------------ Dataset summary tab
function renderSummary(s) {
  const rows = ["train", "validation", "test", "total"].map((k) => ({ split: k[0].toUpperCase() + k.slice(1), ...s[k] }));
  const n = (v) => v.toLocaleString("en-US");
  renderTable($("summary-table"), [
    { key: "split", label: "Split" }, { key: "sequences", label: "Sequences", num: true },
    { key: "frames", label: "Frames", num: true, fmt: n }, { key: "actor_tracks", label: "Actor tracks", num: true },
    { key: "two_actor_seqs", label: "Two-actor sequences", num: true },
    { key: "valid_person_frames", label: "Valid person-frames", num: true, fmt: n },
    { key: "occ_seqs", label: "OCC sequences", num: true },
  ], rows);
}

// ------------------------------------------------------------------ the viewer
async function viewer() {
  if (!state.viewer) {
    const v = new WebViewer();
    await v.start(null, $("viewer"), {
      hideWelcomeScreen: true, width: "100%", height: "100%", theme: "dark",
      checkForUpdatesOnStartup: false, allowFullscreen: true,
      renderBackend: "webgl",  // the tested path; WebGPU support differs between browsers
    });
    state.viewer = v;
  }
  return state.viewer;
}
function setStatus(text) { $("status").textContent = text || ""; $("status").hidden = !text; }

async function openSequence(name, frame) {
  const r = state.byName.get(name);
  if (!r) return;
  history.replaceState(null, "", `#seq=${encodeURIComponent(name)}${frame ? `&frame=${frame}` : ""}`);
  state.open = name; showMeta(r);
  for (const t of document.querySelectorAll("table tr.open")) t.classList.remove("open");
  for (const t of document.querySelectorAll(`table tr[data-name="${CSS.escape(name)}"]`)) t.classList.add("open");
  $("empty").hidden = true;
  $("controls").hidden = false;
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (!r.recording) { setStatus(`No recording for ${name} yet. Run: python -m vidHMR_viewer.recording --only ${name}`); return; }

  const mb = `${Math.round(r.recording_mb)} MB`;
  setStatus(`Loading ${name} (${mb}) ...`);
  const v = await viewer();
  if (state.open !== name) return;  // another sequence was selected meanwhile
  try { await v.close("all"); } catch {}  // free the memory of the previous recording
  const url = new URL(r.recording, location.href).href;
  const t0 = performance.now();
  await v.open(url);  // resolves after the first data, or after about 10 s
  if (state.open !== name) return;
  const start = frame ?? (r.pc_range.length ? r.pc_range[0] : 0);
  try { await v.setTime({ timeline: "frame", sequence: start, playState: "paused" }); } catch {}
  diag("opened", { name, ms: Math.round(performance.now() - t0) });

  // The rest of the file loads in the background; the frames that have arrived can play.
  setStatus(`Loading ${name} (${mb}): the first frames can play now`);
  while (state.open === name) {
    const st = await v.getState().catch(() => null);
    if (st && !(st.loading || []).length) break;
    await new Promise((res) => setTimeout(res, 500));
  }
  if (state.open === name) { setStatus(""); diag("loaded", { name, ms: Math.round(performance.now() - t0) }); }
}

function showMeta(r) {
  const sam = state.sam3dByName?.get(r.name);
  const parts = [`<b>${r.name}</b>`, r.split, `${r.frames} frames`, actorsHtml(r.genders)];
  if (r.pc_range.length) parts.push(`PC ${r.pc_range[0]}–${r.pc_range[1]}`);
  if (sam && sam.mpjpe !== null) parts.push(`SAM 3D Body MPJPE ${sam.mpjpe.toFixed(1)} mm`);
  $("meta").innerHTML = parts.join('<span class="sep">·</span>');
  $("meta").hidden = false;
}

// ------------------------------------------------------------------ playback controls
// Play / Pause and the frame above the viewer (Rerun has the same controls, small, under the views).
let lastPlay = null;
async function syncControls() {
  if (!state.viewer || !state.open) return;
  const st = await state.viewer.getState().catch(() => null);
  const rec = st?.recordings?.[0];
  if (!rec) return;
  const playing = rec.playState === "playing" || rec.playState === "following";
  if (playing !== lastPlay) {
    lastPlay = playing;
    $("playlabel").textContent = playing ? "Pause" : "Play";
    $("play").setAttribute("aria-label", playing ? "Pause" : "Play");
    $("playicon").setAttribute("d", playing ? "M4 2.5h3v11H4zM9 2.5h3v11H9z" : "M4 2.5v11l9-5.5z");
  }
  const end = rec.timelines?.find((t) => t.name === "frame")?.timeRange?.end;
  const f = rec.currentTime?.time;
  $("frame").textContent = f === undefined ? "" : `Frame ${f}${end !== undefined ? ` / ${end}` : ""}`;
  // The scrub bar covers the whole sequence; data that has not arrived yet shows the last frame.
  $("scrub").max = state.byName.get(state.open).frames - 1;
  if (!scrubbing && f !== undefined) $("scrub").value = f;
}
let scrubbing = false;
$("scrub").addEventListener("pointerdown", () => { scrubbing = true; });
addEventListener("pointerup", () => { scrubbing = false; });
$("scrub").oninput = () => state.viewer?.setTime({ timeline: "frame", sequence: +$("scrub").value, playState: "paused" }).catch(() => {});
$("scrub").onchange = () => { scrubbing = false; };
async function togglePlay() {
  if (!state.viewer) return;
  const st = await state.viewer.getState().catch(() => null);
  const rec = st?.recordings?.[0];
  if (!rec) return;
  const playing = rec.playState === "playing" || rec.playState === "following";
  const end = rec.timelines?.find((t) => t.name === "frame")?.timeRange?.end;
  const opts = { timeline: "frame", playState: playing ? "paused" : "playing" };
  if (!playing && end !== undefined && rec.currentTime?.time >= end) opts.sequence = 0;  // at the end: restart
  await state.viewer.setTime(opts).catch(() => {});
  diag("toggle", { to: opts.playState });
  syncControls();
}
$("play").onclick = togglePlay;
$("restart").onclick = () => state.viewer?.setTime({ timeline: "frame", sequence: 0 }).then(syncControls).catch(() => {});
addEventListener("keydown", (e) => {
  // In the viewer, Rerun handles Space itself.
  if (e.code !== "Space" || !state.open || $("viewer").contains(e.target)) return;
  if (e.target.closest?.("input, select, textarea, button")) return;
  e.preventDefault(); togglePlay();
});
setInterval(syncControls, 250);

// ------------------------------------------------------------------ diagnostics
// Start info, errors and a 2 s heartbeat (with the viewer state) to the server log
// (generated/logs/client.log). A worker beats too: if only the worker beats, the page is frozen.
const diagId = Math.random().toString(36).slice(2, 8), diagT0 = performance.now();
function diag(kind, data = {}) {
  try {
    navigator.sendBeacon("/api/client-log", new Blob([JSON.stringify({ id: diagId, t: Math.round(performance.now() - diagT0), kind, ...data })], { type: "application/json" }));
  } catch {}
}
(() => {
  let gl = "none";
  try {
    const c = document.createElement("canvas").getContext("webgl2");
    const ext = c && c.getExtension("WEBGL_debug_renderer_info");
    gl = c ? (ext ? c.getParameter(ext.UNMASKED_RENDERER_WEBGL) : c.getParameter(c.RENDERER)) : "no WebGL2";
  } catch (e) { gl = "error " + e; }
  diag("start", { ua: navigator.userAgent, gl, webgpu: !!navigator.gpu, dpr: devicePixelRatio, size: [innerWidth, innerHeight] });
  addEventListener("error", (e) => diag("error", { msg: String(e.message), at: `${e.filename}:${e.lineno}` }));
  addEventListener("unhandledrejection", (e) => diag("rejection", { msg: String(e.reason?.stack || e.reason) }));
  const src = `let n = 0; const t = setInterval(() => { fetch("${location.origin}/api/client-log", { method: "POST",
    body: JSON.stringify({ id: "${diagId}", kind: "wbeat", n: n++ }) }); if (n > 900) clearInterval(t); }, 2000);`;
  try { new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" }))); } catch {}
  let last = performance.now(), n = 0;
  const beat = setInterval(async () => {
    const now = performance.now(), gap = Math.round(now - last); last = now;
    let rec = null, loading = null;
    try {
      const st = state.viewer && await state.viewer.getState();
      rec = st?.recordings?.[0]; loading = (st?.loading || []).length;
    } catch {}
    diag("beat", { gap, open: state.open, frame: rec ? Number(rec.currentTime?.time) : null, play: rec?.playState ?? null, loading });
    if (++n > 900) clearInterval(beat);
  }, 2000);
})();

// ------------------------------------------------------------------ start
for (const b of document.querySelectorAll('[role="tab"]')) b.onclick = () => {
  for (const o of document.querySelectorAll('[role="tab"]')) o.setAttribute("aria-selected", o === b);
  for (const p of document.querySelectorAll(".panel")) p.hidden = p.id !== `tab-${b.dataset.tab}`;
};
$("search").oninput = renderSequences;
$("actors").onchange = renderSequences;

const [seqs, sam3d, summary] = await Promise.all(["sequences", "sam3d", "summary"].map((k) => fetch(`/api/${k}`).then((r) => r.json())));
state.rows = seqs.rows.map((r) => ({
  ...r,
  tags: r.subsets.filter((t) => t === "occ" || t === "pc").map((t) => (t === "occ" ? "3DPW-OCC" : "3DPW-PC")).join(", "),
  pc: r.pc_range.length ? `${r.pc_range[0]}–${r.pc_range[1]}` : "",
}));
state.subsets = seqs.subsets;
state.byName = new Map(state.rows.map((r) => [r.name, r]));
state.sam3dByName = new Map((sam3d?.sequences || []).map((r) => [r.name, r]));
state.sam3dWorst = Object.fromEntries((sam3d?.sequences || []).map((r) => [r.name, r.worst_frame]));
renderSequences(); renderSam3d(sam3d); renderSummary(summary);
state.ready = true;

const h = new URLSearchParams(location.hash.slice(1));
if (h.get("seq")) openSequence(h.get("seq"), h.get("frame") ? +h.get("frame") : undefined);
