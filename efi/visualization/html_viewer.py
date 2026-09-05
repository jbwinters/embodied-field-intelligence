"""Self-contained HTML episode viewer.

Ships the raw field data (uint8-quantized per frame, with per-frame and
per-episode scale factors) into a single offline HTML file and renders
client-side on <canvas>:

- crisp nearest-neighbor heatmaps with per-field single-hue ramps;
- EPISODE-global normalization by default -- per-frame autoscale is what
  made the old PNG viewer strobe frame-to-frame -- with a per-frame toggle;
- a probe crosshair synchronized across every panel (hover any panel, read
  the same cell off all of them; click to pin);
- trajectory and policy-distribution overlays on the world panel (the
  policy is recomputed client-side from V and lambda: pi ∝ exp(V/lambda));
- telemetry strips (reward, lambda, residual, valences, affect) with a
  shared playhead; click or drag to seek;
- keyboard transport (space, arrows, shift+arrows, home/end).

No external assets, fonts, or CDNs: the file works offline and in an
archive. Colors are the validated dark-mode categorical palette from the
data-viz reference (all slots >= 3:1 on the surface, worst adjacent CVD
dE 23.7); each field/series wears its entity's hue everywhere it appears.
"""

import base64
import json
from pathlib import Path

import numpy as np

# Entity -> hue (validated dark-surface categorical palette). Cost-family
# quantities (q, pain, membrane) deliberately share the red family: they are
# literally components of the state cost.
FIELD_SPECS = [
    # key, lmdp label, legacy label, hex hue
    ("GA",        "p(A) belief",      "A scent",         "#199e70"),
    ("GB",        "p(B) belief",      "B scent",         "#9085e9"),
    ("P_eff",     "V · value",   "potential",       "#c98500"),
    ("Qcost",     "q · state costs", "q · state costs", "#e66767"),
    ("Epistemic", "info gain",        "frontier pull",   "#3987e5"),
    ("Vtrail",    "trail",            "trail",           "#d95926"),
    ("Novel",     "novelty",          "novelty",         "#d55181"),
    ("Ssum",      "schema bias",      "schema bias",     "#7c7f74"),
    ("Pain",      "pain field",       "pain field",      "#e66767"),
    ("Membrane",  "membrane",         "membrane",        "#d95926"),
    ("Goal", "goal · observed value", "goal · observed value", "#199e70"),
    ("Object", "object · observed", "object · observed", "#3987e5"),
    ("ActionValue", "action · predicted value", "action · predicted value", "#c98500"),
    ("Unresolved", "unresolved · probability", "unresolved · probability", "#e66767"),
    ("ObjectNext", "object · predicted next", "object · predicted next", "#3987e5"),
    ("ObjectFuture", "object · forecast horizon", "object · forecast horizon", "#3987e5"),
    ("BodyNext", "body · predicted next", "body · predicted next", "#f2f3ee"),
]

STRIP_SPECS = [
    # id, label, [(info key, series label, hex)], value format digits
    ("reward",   "reward / step", [("reward", "reward", "#199e70")], 3),
    ("lam",      "λ · risk temperature", [("lam", "λ", "#c98500")], 4),
    ("residual", "value residual", [("residual", "residual", "#3987e5")], 4),
    ("valence",  "valences", [("valA", "w(A)", "#199e70"), ("valB", "w(B)", "#9085e9")], 2),
    ("affect",   "affect", [("pain", "pain", "#e66767"), ("arousal", "arousal", "#d95926")], 2),
    ("learning", "empirical support", [("learned_transitions", "effective observations", "#3987e5")], 1),
    ("prediction", "observed effect · log loss", [("prediction_loss", "log loss", "#d55181")], 3),
]


def _b64_bytes(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def _pack_field(frames_list):
    """Quantize a field's frames to uint8 with per-frame scales plus the
    exact episode-global envelope. The envelope (not per-frame autoscale)
    is the display default: it is constant across frames by construction,
    which is what stops the colormap strobing, and unlike a percentile it
    cannot saturate away sparse peaks (a belief field is prior-valued
    almost everywhere, so high percentiles collapse onto the prior)."""
    los, his, packed = [], [], []
    for f in frames_list:
        f = np.asarray(f, dtype=np.float32)
        lo, hi = float(f.min()), float(f.max())
        if hi <= lo:
            q = np.zeros(f.shape, dtype=np.uint8)
            hi = lo + 1.0
        else:
            q = np.clip((f - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        los.append(lo)
        his.append(hi)
        packed.append(_b64_bytes(q))
    ep_lo, ep_hi = min(los), max(his)
    if ep_hi <= ep_lo:
        ep_hi = ep_lo + 1.0
    return {"frames": packed, "lo": los, "hi": his, "eplo": ep_lo, "ephi": ep_hi}


def build_payload(episode_data: dict, final_metrics: dict = None) -> dict:
    frames = episode_data.get("frames", [])
    world_frames = episode_data.get("world_frames", [])
    if not frames:
        return None
    first_key = next(key for key, *_ in FIELD_SPECS if key in frames[0])
    H, W = np.asarray(frames[0][first_key]).shape
    lmdp = "lam" in frames[0].get("info", {})

    fields = []
    for key, lmdp_label, legacy_label, hue in FIELD_SPECS:
        present = [f for f in frames if key in f]
        if len(present) != len(frames):
            continue
        arrs = [f[key] for f in frames]
        if key == "Ssum" and not any(np.any(a) for a in arrs):
            continue  # retired Oja schema: hide the dead panel
        packed = _pack_field(arrs)
        packed["key"] = key
        packed["label"] = lmdp_label if lmdp else legacy_label
        packed["hue"] = hue
        fields.append(packed)

    world = [_b64_bytes(np.asarray(w, dtype=np.uint8)) for w in world_frames]
    walls = None
    if all("Walls" in f for f in frames):
        walls = [_b64_bytes(np.asarray(f["Walls"], dtype=np.uint8)) for f in frames]

    info = [f.get("info", {}) for f in frames]

    fm = dict(final_metrics or {})
    fm = {k: (None if v is None else float(v)) for k, v in fm.items()
          if isinstance(v, (int, float)) or v is None}

    payload = {
        "H": int(H), "W": int(W), "n": len(frames), "lmdp": bool(lmdp),
        "fields": fields, "world": world, "walls": walls, "info": info,
        "final": fm,
    }
    if episode_data.get("title"):
        payload["title"] = str(episode_data["title"])
    for key in ("guide", "presentation"):
        if key in episode_data:
            payload[key] = episode_data[key]
    return payload


# ----------------------------------------------------------------------
CSS = """
:root {
  --bg: #121312; --surface: #1a1a19; --surface2: #222321; --line: #30322d;
  --ink: #f2f3ee; --ink2: #b9bbb0; --ink3: #83867a;
  --accent: #c98500;
  --mono: "JetBrains Mono", "SF Mono", ui-monospace, Menlo, Consolas, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--ink);
  font-family: var(--sans); font-size: 14px; }
body { padding: 14px 18px 40px; }
button, select { font-family: var(--mono); font-size: 12px; }

.nameplate { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
  border-bottom: 1px solid var(--line); padding-bottom: 10px; }
.nameplate h1 { font-family: var(--mono); font-size: 15px; font-weight: 600;
  letter-spacing: 0.22em; margin: 0; }
.nameplate h1 .dim { color: var(--ink3); font-weight: 400; }
.runmeta { font-family: var(--mono); font-size: 11px; color: var(--ink2);
  letter-spacing: 0.06em; }
.finals { margin-left: auto; display: flex; gap: 10px; flex-wrap: wrap; }
.finals span { font-family: var(--mono); font-size: 11px; color: var(--ink3); }
.finals b { color: var(--ink2); font-weight: 600; }

.transport { display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  padding: 10px 0; position: sticky; top: 0; background: var(--bg); z-index: 5;
  border-bottom: 1px solid var(--line); }
.transport button {
  background: var(--surface2); color: var(--ink); border: 1px solid var(--line);
  border-radius: 4px; padding: 5px 10px; cursor: pointer; }
.transport button:hover { border-color: var(--ink3); }
.transport button:focus-visible, .toggle:focus-visible, select:focus-visible,
input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.transport button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }
.transport select { background: var(--surface2); color: var(--ink);
  border: 1px solid var(--line); border-radius: 4px; padding: 5px 6px; }
.transport .sep { width: 1px; height: 20px; background: var(--line); margin: 0 4px; }
.readout { font-family: var(--mono); font-size: 12px; color: var(--ink2);
  margin-left: auto; white-space: nowrap; }
.readout b { color: var(--ink); font-weight: 600; }
#seek { width: 100%; margin: 10px 0 2px; accent-color: var(--accent); }
kbd { font-family: var(--mono); font-size: 10px; color: var(--ink3);
  border: 1px solid var(--line); border-radius: 3px; padding: 0 4px; }
.hints { font-size: 10px; color: var(--ink3); padding: 2px 0 8px;
  font-family: var(--mono); }
.guide { margin: 14px 0; padding: 14px 16px; border: 1px solid var(--line);
  background: var(--surface); border-radius: 6px; }
.guide h2 { font-size: 17px; font-weight: 550; margin: 0 0 7px; }
.guide p { margin: 6px 0; color: var(--ink2); line-height: 1.5; max-width: 100ch; }
.guide .legend { display: flex; flex-wrap: wrap; gap: 8px 18px; margin: 12px 0;
  font-family: var(--mono); font-size: 11px; }
.guide .legend span { display: inline-flex; gap: 6px; align-items: center; }
.guide .legend i { width: 10px; height: 10px; border-radius: 2px; }
.chapters { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.chapters button, .guide a { color: var(--accent); background: var(--surface2);
  border: 1px solid var(--line); padding: 6px 10px; border-radius: 4px; cursor: pointer; }
.chapters button[aria-current="true"] { border-color: var(--accent); }
.narration { margin-top: 12px; }
.narration p { font-size: 12px; line-height: 1.55; margin: 8px 0; color: var(--ink2); }
.narration p:first-of-type { color: var(--accent); font-family: var(--mono); }
.narration p:last-child { color: var(--ink3); font-size: 11px; }

.main { display: grid; grid-template-columns: 380px 1fr; gap: 14px; }
@media (max-width: 980px) { .main { grid-template-columns: 1fr; } }

.panel { background: var(--surface); border: 1px solid var(--line);
  border-radius: 6px; padding: 8px; }
.panel h3 { margin: 0 0 6px; font-family: var(--mono); font-size: 10.5px;
  font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--ink2); display: flex; align-items: center; gap: 6px; }
.panel h3 .chip { width: 8px; height: 8px; border-radius: 2px; flex: none; }
.panel h3 .range { margin-left: auto; color: var(--ink3); font-weight: 400;
  letter-spacing: 0; text-transform: none; }
.panel canvas { width: 100%; image-rendering: pixelated; display: block;
  border-radius: 3px; cursor: crosshair; }

.fieldgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
  gap: 10px; align-content: start; }

.probe { margin-top: 12px; }
.probe table { width: 100%; border-collapse: collapse; font-family: var(--mono);
  font-size: 11.5px; }
.probe td { padding: 3px 4px; border-bottom: 1px solid var(--line); }
.probe td:last-child { text-align: right; color: var(--ink); }
.probe td:first-child { color: var(--ink2); }
.probe .coords { color: var(--ink3); font-size: 10.5px; padding: 2px 0 6px;
  font-family: var(--mono); }
.probe .pinned { color: var(--accent); }

.strips { margin-top: 16px; display: grid; gap: 8px; }
.strip { background: var(--surface); border: 1px solid var(--line);
  border-radius: 6px; padding: 6px 8px 2px; }
.strip .head { display: flex; gap: 10px; align-items: baseline;
  font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--ink2); }
.strip .legend { display: flex; gap: 8px; text-transform: none; letter-spacing: 0; }
.strip .legend span { display: inline-flex; align-items: center; gap: 4px;
  color: var(--ink3); font-size: 10.5px; }
.strip .legend i { width: 10px; height: 2px; display: inline-block; }
.strip .now { margin-left: auto; color: var(--ink); font-size: 11px; }
.strip canvas { width: 100%; height: 54px; display: block; cursor: ew-resize; }

#modal { position: fixed; inset: 0; background: rgba(10,10,10,0.82);
  display: none; align-items: center; justify-content: center; z-index: 50; }
#modal.open { display: flex; }
#modal .box { background: var(--surface); border: 1px solid var(--line);
  border-radius: 8px; padding: 14px; max-width: min(88vmin, 900px); width: 70vmin; }
#modal canvas { width: 100%; image-rendering: pixelated; border-radius: 4px;
  cursor: crosshair; }
#modal .bar { height: 10px; border-radius: 3px; margin-top: 8px; }
#modal .barlabels { display: flex; justify-content: space-between;
  font-family: var(--mono); font-size: 10.5px; color: var(--ink3); }

.tip { position: fixed; pointer-events: none; background: var(--surface2);
  border: 1px solid var(--line); border-radius: 4px; padding: 4px 7px;
  font-family: var(--mono); font-size: 11px; color: var(--ink); z-index: 60;
  display: none; white-space: nowrap; }

@media (prefers-reduced-motion: no-preference) {
  .transport button { transition: border-color 120ms ease, color 120ms ease; }
}
"""

# The JS is written against the payload injected as `DATA`.
JS = r"""
const D = DATA;
const N = D.n, H = D.H, W = D.W;
const SURFACE = [26, 26, 25];

// ---- decode ---------------------------------------------------------
function b64ToBytes(s) {
  const bin = atob(s); const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
for (const f of D.fields) f.data = f.frames.map(b64ToBytes);
D.worldData = D.world.map(b64ToBytes);
D.wallsData = D.walls ? D.walls.map(b64ToBytes) : null;

function fieldValue(f, t, y, x) {
  const b = f.data[t][y * W + x];
  return f.lo[t] + (b / 255) * (f.hi[t] - f.lo[t]);
}

// ---- color ramps ----------------------------------------------------
function hexRGB(h) { return [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)); }
function mix(a, b, t) { return a.map((v, i) => Math.round(v + (b[i] - v) * t)); }
function makeRamp(hex) {
  const hue = hexRGB(hex), tip = mix(hue, [242, 243, 238], 0.72);
  const lut = new Array(256);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    lut[i] = t < 0.62 ? mix(SURFACE, hue, t / 0.62) : mix(hue, tip, (t - 0.62) / 0.38);
  }
  return lut;
}
for (const f of D.fields) f.ramp = makeRamp(f.hue);

// ---- state ----------------------------------------------------------
const S = {
  t: 0, playing: false, timer: null, speed: 1, loop: D.presentation?.loop ?? true,
  scaleMode: 'episode',            // 'episode' kills the per-frame strobing
  showPath: true, showPolicy: D.presentation?.show_policy ?? D.lmdp,
  hover: null,                     // {y, x} synchronized probe cell
  pinned: null,
  modalField: null,
};

// ---- panels ---------------------------------------------------------
const CELL = 14;
const grid = document.getElementById('fieldgrid');
const panels = [];

function makePanel(f) {
  const div = document.createElement('div'); div.className = 'panel';
  const h3 = document.createElement('h3');
  h3.innerHTML = `<span class="chip" style="background:${f.hue}"></span>` +
                 `<button class="zoom" style="all:unset;cursor:zoom-in" ` +
                 `aria-label="Enlarge ${f.label}">${f.label}</button>` +
                 `<span class="range"></span>`;
  const cv = document.createElement('canvas');
  cv.width = W * CELL; cv.height = H * CELL;
  div.appendChild(h3); div.appendChild(cv); grid.appendChild(div);
  const p = { f, cv, ctx: cv.getContext('2d'),
              range: h3.querySelector('.range'),
              img: new ImageData(W, H),
              off: Object.assign(document.createElement('canvas'), {width: W, height: H}) };
  p.offCtx = p.off.getContext('2d');
  h3.querySelector('.zoom').addEventListener('click', () => openModal(f));
  bindProbe(cv, p);
  panels.push(p);
}

function drawField(p, t) {
  const f = p.f, data = f.data[t], px = p.img.data;
  const lo = S.scaleMode === 'episode' ? f.eplo : f.lo[t];
  const hi = S.scaleMode === 'episode' ? f.ephi : f.hi[t];
  const span = hi - lo || 1;
  for (let i = 0; i < H * W; i++) {
    const v = f.lo[t] + (data[i] / 255) * (f.hi[t] - f.lo[t]);
    let u = (v - lo) / span; u = u < 0 ? 0 : u > 1 ? 1 : u;
    const c = f.ramp[(u * 255) | 0], j = i * 4;
    px[j] = c[0]; px[j + 1] = c[1]; px[j + 2] = c[2]; px[j + 3] = 255;
  }
  p.offCtx.putImageData(p.img, 0, 0);
  const ctx = p.ctx;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(p.off, 0, 0, W * CELL, H * CELL);
  if (D.presentation?.field_context) {
    // Context marks are observed walls and current proprioception, not extra field values.
    ctx.fillStyle = 'rgba(130,138,121,0.25)';
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (wallAt(t, y, x)) ctx.fillRect(x*CELL, y*CELL, CELL, CELL);
    }
    const pos = D.info[t].pos;
    if (pos) {
      ctx.strokeStyle = 'rgba(242,243,238,0.65)'; ctx.lineWidth = 1;
      ctx.strokeRect(pos[1]*CELL+1, pos[0]*CELL+1, CELL-2, CELL-2);
    }
  }
  drawCrosshair(ctx);
  p.range.textContent = lo.toFixed(2) + ' → ' + hi.toFixed(2);
}

// ---- world panel ----------------------------------------------------
const worldCv = document.getElementById('world');
worldCv.width = W * CELL; worldCv.height = H * CELL;
const worldCtx = worldCv.getContext('2d');
const worldImg = new ImageData(W, H);
const worldOff = Object.assign(document.createElement('canvas'), {width: W, height: H});
const worldOffCtx = worldOff.getContext('2d');
bindProbe(worldCv, null);

function wallAt(t, y, x) {
  if (!D.wallsData) return false;
  return D.wallsData[t][y * W + x] > 0;
}

function policyAt(t) {
  // A controller with joint consequences supplies its ACTUAL distribution.
  // Preserve the original value-derived fallback for earlier recordings.
  const recorded = D.info[t];
  if (recorded && Object.prototype.hasOwnProperty.call(recorded, 'policy')) {
    if (!recorded.policy || !recorded.pos) return null;
    return {probs: recorded.policy,
            dirs: [[-1,0],[1,0],[0,-1],[0,1],[0,0]], pos: recorded.pos};
  }
  // pi(a) proportional to exp(V(u)/lambda) over open neighbors -- the
  // closed-form LMDP policy, recomputed from the shipped value field.
  const vf = D.fields.find(f => f.key === 'P_eff');
  const info = D.info[t];
  if (!vf || !info || info.lam === undefined || !info.pos) return null;
  const [y, x] = info.pos, lam = Math.max(info.lam, 1e-6);
  const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  const scores = dirs.map(([dy, dx]) => {
    const ny = y + dy, nx = x + dx;
    if (ny < 0 || ny >= H || nx < 0 || nx >= W) return -Infinity;
    if (wallAt(t, ny, nx)) return -Infinity;
    const v = fieldValue(vf, t, ny, nx);
    return v < -1e8 ? -Infinity : v / lam;
  });
  const m = Math.max(...scores);
  if (m === -Infinity) return null;
  const es = scores.map(s => Math.exp(s - m));
  const z = es.reduce((a, b) => a + b, 0);
  return { probs: es.map(e => e / z), dirs, pos: [y, x] };
}

function drawWorld(t) {
  const rgb = D.worldData[t], px = worldImg.data;
  for (let i = 0; i < H * W; i++) {
    px[i * 4] = rgb[i * 3]; px[i * 4 + 1] = rgb[i * 3 + 1];
    px[i * 4 + 2] = rgb[i * 3 + 2]; px[i * 4 + 3] = 255;
  }
  worldOffCtx.putImageData(worldImg, 0, 0);
  worldCtx.imageSmoothingEnabled = false;
  worldCtx.drawImage(worldOff, 0, 0, W * CELL, H * CELL);

  if (S.showPath) {
    worldCtx.strokeStyle = 'rgba(242,243,238,0.55)';
    worldCtx.lineWidth = 2; worldCtx.beginPath();
    for (let k = 0; k <= t; k++) {
      if (D.info[t].scene !== undefined && D.info[k].scene !== D.info[t].scene) continue;
      const pos = D.info[k] && D.info[k].pos; if (!pos) continue;
      const cx = pos[1] * CELL + CELL / 2, cy = pos[0] * CELL + CELL / 2;
      const newScene = k === 0 || D.info[k].scene !== D.info[k - 1].scene;
      newScene ? worldCtx.moveTo(cx, cy) : worldCtx.lineTo(cx, cy);
    }
    worldCtx.stroke();
  }
  const pos = D.info[t] && D.info[t].pos;
  if (pos) {
    worldCtx.strokeStyle = '#f2f3ee'; worldCtx.lineWidth = 2;
    worldCtx.strokeRect(pos[1] * CELL + 1, pos[0] * CELL + 1, CELL - 2, CELL - 2);
  }
  // Optional sensed goal paint can remain visible under an interactive object.
  for (const [y,x] of (D.info[t].goal_markers || [])) {
    worldCtx.strokeStyle = '#199e70'; worldCtx.lineWidth = 2;
    worldCtx.strokeRect(x * CELL + 2, y * CELL + 2, CELL - 4, CELL - 4);
  }
  const radius = D.info[t].sensing_radius;
  if (pos && Number.isFinite(radius)) {
    worldCtx.save(); worldCtx.strokeStyle = 'rgba(242,243,238,0.5)';
    worldCtx.setLineDash([3,3]); worldCtx.lineWidth = 0.8;
    worldCtx.strokeRect((pos[1]-radius)*CELL, (pos[0]-radius)*CELL,
                       (2*radius+1)*CELL, (2*radius+1)*CELL);
    worldCtx.restore();
  }
  for (const marker of (D.info[t].markers || [])) {
    worldCtx.font = 'bold 9px monospace'; worldCtx.textAlign = 'center';
    worldCtx.textBaseline = 'middle'; worldCtx.fillStyle = marker.color || '#f2f3ee';
    worldCtx.fillText(marker.text, (marker.pos[1]+0.5)*CELL, (marker.pos[0]+0.5)*CELL);
  }
  if (S.showPolicy) {
    const pol = policyAt(t);
    if (pol) {
      const [y, x] = pol.pos;
      const cx = x * CELL + CELL / 2, cy = y * CELL + CELL / 2;
      pol.dirs.forEach(([dy, dx], i) => {
        const p = pol.probs[i]; if (p < 0.01) return;
        const len = CELL * (0.5 + 2.2 * p);
        worldCtx.strokeStyle = '#c98500'; worldCtx.lineWidth = 2.5;
        if (dy === 0 && dx === 0) {
          worldCtx.beginPath(); worldCtx.arc(cx, cy, 2 + 3*p, 0, 2*Math.PI);
          worldCtx.stroke(); return;
        }
        worldCtx.beginPath(); worldCtx.moveTo(cx, cy);
        worldCtx.lineTo(cx + dx * len, cy + dy * len); worldCtx.stroke();
      });
    }
  }
  const action = D.info[t].action;
  if (pos && Number.isInteger(action) && action >= 0 && action < 5) {
    const [dy,dx] = [[-1,0],[1,0],[0,-1],[0,1],[0,0]][action];
    const cx = (pos[1]+0.5)*CELL, cy = (pos[0]+0.5)*CELL;
    worldCtx.strokeStyle = '#c98500'; worldCtx.lineWidth = 2;
    worldCtx.beginPath();
    if (action === 4) worldCtx.arc(cx, cy, CELL*0.37, 0, 2*Math.PI);
    else {
      const ex = cx+dx*CELL*0.95, ey = cy+dy*CELL*0.95;
      worldCtx.moveTo(cx+dx*CELL*0.45, cy+dy*CELL*0.45); worldCtx.lineTo(ex,ey);
      worldCtx.moveTo(ex-dx*4+dy*3, ey-dy*4-dx*3); worldCtx.lineTo(ex,ey);
      worldCtx.lineTo(ex-dx*4-dy*3, ey-dy*4+dx*3);
    }
    worldCtx.stroke();
  }
  drawCrosshair(worldCtx);
}

function drawCrosshair(ctx) {
  const c = S.pinned || S.hover; if (!c) return;
  ctx.strokeStyle = 'rgba(242,243,238,0.35)'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(c.x * CELL + CELL / 2, 0); ctx.lineTo(c.x * CELL + CELL / 2, H * CELL);
  ctx.moveTo(0, c.y * CELL + CELL / 2); ctx.lineTo(W * CELL, c.y * CELL + CELL / 2);
  ctx.stroke();
  ctx.strokeStyle = S.pinned ? '#c98500' : 'rgba(242,243,238,0.9)';
  ctx.strokeRect(c.x * CELL + 0.5, c.y * CELL + 0.5, CELL - 1, CELL - 1);
}

// ---- probe ----------------------------------------------------------
function bindProbe(cv, _p) {
  cv.addEventListener('mousemove', e => {
    const r = cv.getBoundingClientRect();
    const x = Math.floor((e.clientX - r.left) / r.width * W);
    const y = Math.floor((e.clientY - r.top) / r.height * H);
    if (x < 0 || x >= W || y < 0 || y >= H) return;
    if (!S.hover || S.hover.x !== x || S.hover.y !== y) { S.hover = {y, x}; render(); }
  });
  cv.addEventListener('mouseleave', () => { S.hover = null; render(); });
  cv.addEventListener('click', e => {
    const r = cv.getBoundingClientRect();
    const x = Math.floor((e.clientX - r.left) / r.width * W);
    const y = Math.floor((e.clientY - r.top) / r.height * H);
    S.pinned = (S.pinned && S.pinned.x === x && S.pinned.y === y) ? null : {y, x};
    render();
  });
}

function renderProbe() {
  const c = S.pinned || S.hover;
  const coords = document.getElementById('probeCoords');
  const tbody = document.getElementById('probeBody');
  if (!c) {
    coords.textContent = 'hover a panel · click to pin';
    tbody.innerHTML = '';
    return;
  }
  coords.innerHTML = `cell (${c.y}, ${c.x})` +
    (S.pinned ? ' <span class="pinned">◉ pinned</span>' : '') +
    (wallAt(S.t, c.y, c.x) ? ' · known wall' : '');
  tbody.innerHTML = D.fields.map(f => {
    const v = fieldValue(f, S.t, c.y, c.x);
    return `<tr><td><span class="chip" style="display:inline-block;width:8px;height:8px;` +
           `border-radius:2px;background:${f.hue};margin-right:6px"></span>${f.label}</td>` +
           `<td>${v.toFixed(4)}</td></tr>`;
  }).join('');
}

// ---- strips ---------------------------------------------------------
const strips = [];
function makeStrips() {
  const host = document.getElementById('strips');
  for (const spec of STRIPS) {
    const series = spec.series
      .filter(s => D.info.some(i => i[s.key] !== undefined))
      .map(s => ({...s, vals: D.info.map(i => i[s.key])}));
    if (!series.length) continue;
    const div = document.createElement('div'); div.className = 'strip';
    const legend = series.length > 1
      ? `<span class="legend">` + series.map(s =>
          `<span><i style="background:${s.hex}"></i>${s.label}</span>`).join('') + `</span>`
      : '';
    div.innerHTML = `<div class="head">${spec.label}${legend}` +
                    `<span class="now"></span></div>`;
    const cv = document.createElement('canvas');
    div.appendChild(cv); host.appendChild(div);
    const st = { spec, series, cv, ctx: cv.getContext('2d'),
                 now: div.querySelector('.now'), hoverT: null };
    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (const v of s.vals) {
      if (v === undefined || v === null) continue;
      if (v < lo) lo = v; if (v > hi) hi = v;
    }
    if (hi <= lo) hi = lo + 1;
    st.lo = lo; st.hi = hi;
    bindStrip(st);
    strips.push(st);
  }
}

function bindStrip(st) {
  const seekTo = e => {
    const r = st.cv.getBoundingClientRect();
    const t = Math.round((e.clientX - r.left) / r.width * (N - 1));
    setFrame(Math.max(0, Math.min(N - 1, t)));
  };
  let down = false;
  st.cv.addEventListener('mousedown', e => { down = true; stopPlay(); seekTo(e); });
  window.addEventListener('mousemove', e => { if (down) seekTo(e); });
  window.addEventListener('mouseup', () => { down = false; });
  st.cv.addEventListener('mousemove', e => {
    const r = st.cv.getBoundingClientRect();
    st.hoverT = Math.round((e.clientX - r.left) / r.width * (N - 1));
    showTip(e, st); drawStrip(st);
  });
  st.cv.addEventListener('mouseleave', () => { st.hoverT = null; hideTip(); drawStrip(st); });
}

function drawStrip(st) {
  const cv = st.cv, dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight || 54;
  if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
  const ctx = st.ctx; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const X = t => t / (N - 1) * (w - 2) + 1;
  const Y = v => h - 4 - (v - st.lo) / (st.hi - st.lo) * (h - 10);
  // recessive zero line if in range
  if (st.lo < 0 && st.hi > 0) {
    ctx.strokeStyle = 'rgba(131,134,122,0.35)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(w, Y(0)); ctx.stroke();
  }
  for (const s of st.series) {
    ctx.strokeStyle = s.hex; ctx.lineWidth = 2; ctx.beginPath();
    let started = false;
    for (let t = 0; t < N; t++) {
      const scene = D.info[t] && D.info[t].scene;
      if (scene !== undefined && (t === 0 || scene !== D.info[t-1].scene)) started = false;
      const v = s.vals[t];
      if (v === undefined || v === null) { if (scene !== undefined) started = false; continue; }
      started ? ctx.lineTo(X(t), Y(v)) : ctx.moveTo(X(t), Y(v)); started = true;
      if (scene !== undefined) { ctx.fillStyle = s.hex; ctx.fillRect(X(t)-1, Y(v)-1, 2, 2); }
    }
    ctx.stroke();
  }
  // playhead
  ctx.strokeStyle = '#f2f3ee'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(X(S.t), 0); ctx.lineTo(X(S.t), h); ctx.stroke();
  if (st.hoverT !== null) {
    ctx.strokeStyle = 'rgba(242,243,238,0.3)';
    ctx.beginPath(); ctx.moveTo(X(st.hoverT), 0); ctx.lineTo(X(st.hoverT), h); ctx.stroke();
  }
  const d = st.spec.digits;
  st.now.textContent = st.series.map(s => {
    const v = s.vals[S.t];
    return v === undefined || v === null ? '—' : v.toFixed(d);
  }).join(' · ');
}

const tip = document.getElementById('tip');
function showTip(e, st) {
  if (st.hoverT === null) return;
  const d = st.spec.digits;
  tip.innerHTML = `t=${st.hoverT} · ` + st.series.map(s => {
    const v = s.vals[st.hoverT];
    return `${s.label} ${v === undefined || v === null ? '—' : v.toFixed(d)}`;
  }).join(' · ');
  tip.style.display = 'block';
  tip.style.left = Math.min(e.clientX + 12, window.innerWidth - 180) + 'px';
  tip.style.top = (e.clientY - 30) + 'px';
}
function hideTip() { tip.style.display = 'none'; }

// ---- modal ----------------------------------------------------------
const modal = document.getElementById('modal');
const modalCv = document.getElementById('modalCanvas');
const modalCtx = modalCv.getContext('2d');
bindProbe(modalCv, null);  // bind once; openModal must not stack listeners
function openModal(f) {
  S.modalField = f;
  document.getElementById('modalTitle').innerHTML =
    `<span class="chip" style="background:${f.hue}"></span>${f.label}`;
  const bar = document.getElementById('modalBar');
  const stops = [0, 0.25, 0.5, 0.75, 1].map(t => {
    const c = f.ramp[(t * 255) | 0]; return `rgb(${c[0]},${c[1]},${c[2]}) ${t * 100}%`;
  });
  bar.style.background = `linear-gradient(90deg, ${stops.join(',')})`;
  modal.classList.add('open');
  render();
}
document.getElementById('modalClose').addEventListener('click', () => {
  modal.classList.remove('open'); S.modalField = null;
});
modal.addEventListener('click', e => {
  if (e.target === modal) { modal.classList.remove('open'); S.modalField = null; }
});

function drawModal() {
  const f = S.modalField; if (!f) return;
  const t = S.t, img = new ImageData(W, H), px = img.data;
  const lo = S.scaleMode === 'episode' ? f.eplo : f.lo[t];
  const hi = S.scaleMode === 'episode' ? f.ephi : f.hi[t];
  const span = hi - lo || 1;
  for (let i = 0; i < H * W; i++) {
    const v = f.lo[t] + (f.data[t][i] / 255) * (f.hi[t] - f.lo[t]);
    let u = (v - lo) / span; u = u < 0 ? 0 : u > 1 ? 1 : u;
    const c = f.ramp[(u * 255) | 0], j = i * 4;
    px[j] = c[0]; px[j + 1] = c[1]; px[j + 2] = c[2]; px[j + 3] = 255;
  }
  const off = Object.assign(document.createElement('canvas'), {width: W, height: H});
  off.getContext('2d').putImageData(img, 0, 0);
  modalCv.width = W * 24; modalCv.height = H * 24;
  modalCtx.imageSmoothingEnabled = false;
  modalCtx.drawImage(off, 0, 0, W * 24, H * 24);
  const c = S.pinned || S.hover;
  if (c) {
    modalCtx.strokeStyle = 'rgba(242,243,238,0.8)';
    modalCtx.strokeRect(c.x * 24 + 0.5, c.y * 24 + 0.5, 23, 23);
  }
  document.getElementById('modalLo').textContent = lo.toFixed(3);
  document.getElementById('modalHi').textContent = hi.toFixed(3);
}

// ---- transport ------------------------------------------------------
function setFrame(t) { S.t = t; render(); }
function startPlay() {
  if (S.playing) return;
  S.playing = true;
  document.getElementById('play').textContent = 'Pause';
  document.getElementById('play').setAttribute('aria-pressed', 'true');
  S.timer = setInterval(() => {
    if (S.t >= N - 1) { S.loop ? setFrame(0) : stopPlay(); return; }
    setFrame(S.t + 1);
  }, 1000 / ((D.presentation?.fps || 8) * S.speed));
}
function stopPlay() {
  S.playing = false;
  document.getElementById('play').textContent = 'Play';
  document.getElementById('play').setAttribute('aria-pressed', 'false');
  clearInterval(S.timer);
}
document.getElementById('play').addEventListener('click',
  () => S.playing ? stopPlay() : startPlay());
for (const [id, dt] of [['b10', -10], ['b1', -1], ['f1', 1], ['f10', 10]]) {
  document.getElementById(id).addEventListener('click', () => {
    stopPlay(); setFrame(Math.max(0, Math.min(N - 1, S.t + dt)));
  });
}
document.getElementById('loop').addEventListener('click', e => {
  S.loop = !S.loop; e.target.setAttribute('aria-pressed', String(S.loop));
});
document.getElementById('speed').addEventListener('change', e => {
  S.speed = parseFloat(e.target.value);
  if (S.playing) { stopPlay(); startPlay(); }
});
document.getElementById('scale').addEventListener('change', e => {
  S.scaleMode = e.target.value; render();
});
document.getElementById('path').addEventListener('click', e => {
  S.showPath = !S.showPath; e.target.setAttribute('aria-pressed', String(S.showPath)); render();
});
const policyBtn = document.getElementById('policy');
policyBtn.setAttribute('aria-pressed', String(S.showPolicy));
document.getElementById('loop').setAttribute('aria-pressed', String(S.loop));
if (D.lmdp) {
  policyBtn.addEventListener('click', e => {
    S.showPolicy = !S.showPolicy;
    e.target.setAttribute('aria-pressed', String(S.showPolicy)); render();
  });
} else policyBtn.style.display = 'none';
document.getElementById('seek').addEventListener('input', e => {
  stopPlay(); setFrame(parseInt(e.target.value));
});

window.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  const step = e.shiftKey ? 10 : 1;
  if (e.code === 'Space') { e.preventDefault(); S.playing ? stopPlay() : startPlay(); }
  else if (e.key === 'ArrowRight') { stopPlay(); setFrame(Math.min(N - 1, S.t + step)); }
  else if (e.key === 'ArrowLeft') { stopPlay(); setFrame(Math.max(0, S.t - step)); }
  else if (e.key === 'Home') { stopPlay(); setFrame(0); }
  else if (e.key === 'End') { stopPlay(); setFrame(N - 1); }
  else if (e.key === 'Escape') { modal.classList.remove('open'); S.modalField = null; }
});

// ---- render ---------------------------------------------------------
function render() {
  const info = D.info[S.t] || {};
  const caption = document.getElementById('episodeCaption');
  caption.hidden = !info.caption;
  caption.textContent = info.caption || '';
  const narration = document.getElementById('narration');
  narration.hidden = !info.narration;
  for (const key of ['next', 'feedback', 'learning']) {
    document.getElementById('narration-' + key).textContent = info.narration?.[key] || '';
  }
  const chapters = D.guide?.chapters || [];
  document.querySelectorAll('#chapters button').forEach((button, i) => {
    button.setAttribute('aria-current', String(S.t >= chapters[i].frame &&
      (i+1 === chapters.length || S.t < chapters[i+1].frame)));
  });
  drawWorld(S.t);
  for (const p of panels) drawField(p, S.t);
  for (const st of strips) drawStrip(st);
  drawModal();
  renderProbe();
  document.getElementById('seek').value = S.t;
  let r = (info.scene !== undefined
          ? `frame <b>${S.t + 1}</b>/${N} · tick ${info.step ?? 0}`
          : `step <b>${String(info.step ?? S.t).padStart(3, '0')}</b>/${N}`) +
          ` · R <b>${(info.return ?? 0).toFixed(3)}</b>`;
  if (info.lam !== undefined) r += ` · λ <b>${info.lam.toFixed(4)}</b>`;
  if (info.residual !== undefined) r += ` · resid <b>${info.residual.toFixed(3)}</b>`;
  document.getElementById('readout').innerHTML = r;
}

// ---- boot -----------------------------------------------------------
document.getElementById('runmeta').textContent =
  `${H}×${W} · ${N} ${D.info[0].scene !== undefined ? 'frames' : 'steps'} · ` +
  (D.fields.some(f => f.key === 'ActionValue') ? 'learned interaction fields' :
    (D.lmdp ? 'value-recursion controller' : 'potential controller'));
if (D.title) document.title = D.title;
if (D.guide) {
  document.getElementById('guide').hidden = false;
  document.getElementById('guideTitle').textContent = D.guide.title || '';
  document.getElementById('guideDescription').textContent = D.guide.description || '';
  document.getElementById('guideNote').textContent = D.guide.note || '';
  for (const item of (D.guide.legend || [])) {
    const span = document.createElement('span'), chip = document.createElement('i');
    chip.style.background = item.color;
    span.append(chip, document.createTextNode(item.label));
    document.getElementById('guideLegend').appendChild(span);
  }
  for (const item of (D.guide.chapters || [])) {
    const button = document.createElement('button'); button.textContent = item.label;
    button.addEventListener('click', () => { stopPlay(); setFrame(item.frame); });
    document.getElementById('chapters').appendChild(button);
  }
  for (const item of (D.guide.links || [])) {
    // Guide links are local sibling replays; reject scripts and remote targets.
    if (!/^[a-zA-Z0-9_-]+\.html(?:#t=\d+)?$/.test(item.href)) continue;
    const link = document.createElement('a'); link.href = item.href; link.textContent = item.label;
    document.getElementById('chapters').appendChild(link);
  }
}
const finals = document.getElementById('finals');
const FINAL_LABELS = {coverage: 'coverage', total_return: 'return',
  targets_A: 'A', targets_B: 'B', bumps_per_100: 'bumps/100',
  mean_cosine: 'alignment', steps: 'steps'};
for (const [k, label] of Object.entries(FINAL_LABELS)) {
  if (D.final && D.final[k] !== undefined && D.final[k] !== null) {
    const v = D.final[k];
    const txt = k === 'coverage' ? (v * 100).toFixed(1) + '%'
      : Number.isInteger(v) ? v : v.toFixed(2);
    finals.insertAdjacentHTML('beforeend', `<span>${label} <b>${txt}</b></span>`);
  }
}
document.getElementById('seek').max = N - 1;
for (const f of D.fields) makePanel(f);
makeStrips();
// Deep-linkable frame: open viewer.html#t=42 at step 42
const hashT = parseInt((location.hash.match(/t=(\d+)/) || [])[1]);
if (!isNaN(hashT)) S.t = Math.max(0, Math.min(N - 1, hashT));
render();
"""

HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EFI episode viewer</title>
<style>__CSS__</style>
</head>
<body>
  <header class="nameplate">
    <h1>EFI <span class="dim">· EPISODE VIEWER</span></h1>
    <span class="runmeta" id="runmeta"></span>
    <span class="finals" id="finals"></span>
  </header>
  <section class="guide" id="guide" hidden>
    <h2 id="guideTitle"></h2><p id="guideDescription"></p>
    <div class="legend" id="guideLegend"></div>
    <p id="guideNote"></p><div class="chapters" id="chapters"></div>
  </section>
  <div id="episodeCaption" hidden style="margin:10px 0;color:var(--ink2)"></div>

  <div class="transport" role="toolbar" aria-label="Playback">
    <button id="play" aria-pressed="false">Play</button>
    <button id="b10" aria-label="Back 10 steps">−10</button>
    <button id="b1" aria-label="Back 1 step">−1</button>
    <button id="f1" aria-label="Forward 1 step">+1</button>
    <button id="f10" aria-label="Forward 10 steps">+10</button>
    <button id="loop" aria-pressed="true">Loop</button>
    <select id="speed" aria-label="Playback speed">
      <option value="0.5">0.5×</option>
      <option value="1" selected>1×</option>
      <option value="2">2×</option>
      <option value="4">4×</option>
    </select>
    <span class="sep"></span>
    <select id="scale" aria-label="Color scale">
      <option value="episode" selected>scale: episode</option>
      <option value="frame">scale: frame</option>
    </select>
    <button id="path" aria-pressed="true">Path</button>
    <button id="policy" aria-pressed="true">Policy π</button>
    <span class="readout" id="readout"></span>
  </div>
  <input type="range" id="seek" min="0" max="0" value="0" aria-label="Seek">
  <div class="hints">
    <kbd>space</kbd> play · <kbd>←</kbd><kbd>→</kbd> step
    · <kbd>shift</kbd>+<kbd>←→</kbd> ×10
    · <kbd>home</kbd>/<kbd>end</kbd> · click a cell to pin the probe
    · click a panel title to enlarge
  </div>

  <div class="main">
    <div>
      <div class="panel">
        <h3><span class="chip" style="background:#f2f3ee"></span>World</h3>
        <canvas id="world"></canvas>
      </div>
      <div class="panel narration" id="narration" hidden>
        <h3>Follow this step</h3>
        <p id="narration-next"></p><p id="narration-feedback"></p>
        <p id="narration-learning"></p>
      </div>
      <div class="panel probe">
        <h3><span class="chip" style="background:#c98500"></span>Probe</h3>
        <div class="coords" id="probeCoords"></div>
        <table><tbody id="probeBody"></tbody></table>
      </div>
    </div>
    <div class="fieldgrid" id="fieldgrid"></div>
  </div>

  <div class="strips" id="strips"></div>

  <div id="modal" role="dialog" aria-modal="true">
    <div class="box">
      <h3 style="font-family:var(--mono);font-size:12px;letter-spacing:0.12em;
                 text-transform:uppercase;color:var(--ink2);display:flex;
                 align-items:center;gap:8px;margin:0 0 8px">
        <span id="modalTitle" style="display:flex;align-items:center;gap:8px"></span>
        <button id="modalClose" style="all:unset;cursor:pointer;margin-left:auto;
                color:var(--ink3)" aria-label="Close">✕ close</button>
      </h3>
      <canvas id="modalCanvas"></canvas>
      <div class="bar" id="modalBar"></div>
      <div class="barlabels"><span id="modalLo"></span><span id="modalHi"></span></div>
    </div>
  </div>

  <div class="tip" id="tip"></div>

<script>
const DATA = __DATA__;
const STRIPS = __STRIPS__;
__JS__
</script>
</body>
</html>
"""


def create_html_viewer(episode_data: dict, output_path: str = "interactive_viewer.html",
                       final_metrics: dict = None) -> str:
    """Build the self-contained viewer HTML from recorded episode data."""
    metrics = episode_data.get("metrics")
    if final_metrics is None and metrics is not None:
        m = vars(metrics) if hasattr(metrics, "__dict__") else metrics
        if isinstance(m, dict):
            final_metrics = {k: m.get(k) for k in
                             ("total_return", "steps", "coverage", "bumps_per_100",
                              "mean_cosine")}
            tc = m.get("targets_collected") or {}
            final_metrics["targets_A"] = tc.get("A")
            final_metrics["targets_B"] = tc.get("B")

    payload = build_payload(episode_data, final_metrics)
    if payload is None:
        print("No frames to display")
        return None

    strips = [{"id": sid, "label": label, "digits": digits,
               "series": [{"key": k, "label": sl, "hex": hx} for k, sl, hx in series]}
              for sid, label, series, digits in STRIP_SPECS]

    html = (HTML_SHELL
            .replace("__CSS__", CSS)
            .replace("__DATA__", json.dumps(payload, separators=(",", ":")))
            .replace("__STRIPS__", json.dumps(strips, separators=(",", ":")))
            .replace("__JS__", JS))

    output_path = Path(output_path)
    output_path.write_text(html)
    return str(output_path.absolute())


def save_episode_as_html(episode_data: dict, output_dir: str = "runs") -> str:
    """Save episode data as an HTML viewer file in `output_dir`."""
    from ..core import ensure_dir, ts

    output_dir = ensure_dir(output_dir)
    return create_html_viewer(episode_data, str(output_dir / f"interactive_episode_{ts()}.html"))
