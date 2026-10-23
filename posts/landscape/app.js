// The terrain around a fixed model while it reads its own reasoning. Height is the log-odds of the right answer, for
// models nudged along two fixed directions in weight space; the sea sits at log-odds zero.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = s => document.querySelector(s);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const C = { bg: css('--bg'), ink: css('--ink'), slate: css('--slate') };
const dpr = Math.min(devicePixelRatio, 2);
const Q = new URLSearchParams(location.search), SHOT = Q.has('shot');   // ?shot: the figure alone, still, without the glass, for stills
document.body.toggleAttribute('data-shot', SHOT);
const DATA = (await (await fetch('./data.json')).json()).traces, R = 3, UP = SHOT ? 241 : 121;

// ------------------------------------------------------------------------------------------- upsampling
const cubic = (p0, p1, p2, p3, t) => 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
function upsample(grid) { // Catmull-Rom over an n×n grid to UP×UP
  const n = grid.length, at = (i, j) => grid[Math.min(n - 1, Math.max(0, i))][Math.min(n - 1, Math.max(0, j))];
  const rows = Array.from({ length: n }, (_, i) => Float64Array.from({ length: UP }, (_, k) => {
    const x = k / (UP - 1) * (n - 1), j = Math.min(Math.floor(x), n - 2), t = x - j;
    return cubic(at(i, j - 1), at(i, j), at(i, j + 1), at(i, j + 2), t);
  }));
  const out = new Float64Array(UP * UP);
  for (let k = 0; k < UP; k++) {
    const y = k / (UP - 1) * (n - 1), i = Math.min(Math.floor(y), n - 2), t = y - i;
    for (let c = 0; c < UP; c++) {
      const r = q => rows[Math.min(n - 1, Math.max(0, q))][c];
      out[k * UP + c] = cubic(r(i - 1), r(i), r(i + 1), r(i + 2), t);
    }
  }
  return out;
}
/** Everything the page needs for one trace: its slices, smoothed, and whether each copy still answers at all. */
function prepare(tr) {
  const F = tr.frames;
  const fields = F.map(f => upsample(f.surface));
  const alive = F.map(f => upsample(f.alive || f.surface.map(r => r.map(() => 0))));
  let mmax = 0; for (const f of fields) for (const v of f) mmax = Math.max(mmax, Math.abs(v));
  let zlo = Infinity, zhi = -Infinity; for (const f of fields) for (const v of f) { zlo = Math.min(zlo, v); zhi = Math.max(zhi, v); }
  return { ...tr, F, fields, alive, HS: 1.25 / mmax, mid: (Math.max(zlo, -25) + Math.min(zhi, 21)) / 2, fr: F.map(f => f.fraction), ends: F.map(f => f.end) };
}
const TRACES = DATA.map(prepare);
if (Q.get('z')) for (const tr of TRACES) tr.HS *= Number(Q.get('z'));   // stills: exaggerate the relief
let T = TRACES[0];

// ------------------------------------------------------------------------------------------- scene
const box = $('#relief');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(dpr);
box.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(26, 1, 0.1, 100);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0.1);
controls.enableDamping = true; controls.enablePan = false; controls.enableZoom = false; controls.maxPolarAngle = Math.PI * 0.47;  // the wheel scrolls the page
let sway = !SHOT;
controls.addEventListener('start', () => { sway = false; });
const CAM = { r: Number(Q.get('r') || 14.4), el: Number(Q.get('el') || 37) * Math.PI / 180, az: Number(Q.get('az') || -105) * Math.PI / 180 };  // high enough to read the island from above, low enough to keep the relief
function placeCamera(time) {
  const az = CAM.az + 0.25 * Math.sin(time * 1e-4), t = controls.target;
  camera.position.set(t.x + CAM.r * Math.cos(CAM.el) * Math.cos(az), t.y + CAM.r * Math.cos(CAM.el) * Math.sin(az), t.z + CAM.r * Math.sin(CAM.el));
  camera.lookAt(t);
}
placeCamera(0);

const hex = h => new THREE.Color(h);
const terrainMat = new THREE.ShaderMaterial({
  side: THREE.DoubleSide,
  uniforms: { paper: { value: hex(C.bg) }, warm: { value: hex('#f3cdb5') }, deep: { value: hex('#d9622f') }, ink: { value: hex(C.ink) },
              under: { value: hex('#9fb6d3') }, light: { value: new THREE.Vector3(-0.5, -0.4, 0.8).normalize() }, hs: { value: T.HS }, disc: { value: R },
              abyss: { value: hex('#4f6e99') }, ramp: { value: 0 } },
  vertexShader: `attribute float alive; varying float vZ; varying vec3 vN; varying vec2 vXY; varying float vAlive;
    void main() { vAlive = alive; vZ = position.z; vN = normal; vXY = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `uniform vec3 paper, warm, deep, ink, under, light, abyss; uniform float hs, disc, ramp; varying float vZ; varying vec3 vN; varying vec2 vXY; varying float vAlive;
    float hash(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
    void main() {
      float rr = length(vXY);
      if (rr > disc) discard;
      float m = vZ / hs;                                   // the log-odds
      float k = m / 2.0;                                   // a contour every 2 units of log-odds
      float w = fwidth(k), d = abs(fract(k + 0.5) - 0.5) / max(w, 1e-4);
      float line = (1.0 - smoothstep(0.5, 1.5, d)) * (1.0 - smoothstep(0.3, 0.8, w)) * 0.45;
      float wc = fwidth(m), coast = 1.0 - smoothstep(0.6, 1.8, abs(m) / max(wc, 1e-4));   // the coastline, margin zero
      float t = clamp(m / 12.0, 0.0, 1.0);
      vec3 land = ramp > 0.5 ? mix(warm, deep, 0.7 * smoothstep(0.0, 1.0, t)) : mix(deep, warm, smoothstep(0.0, 1.0, t));   // red, capped short of the deep colour
      vec3 lost = mix(under, abyss, clamp(-m / 25.0, 0.0, 1.0));   // blue, deeper where less likely
      vec3 base = m > 0.0 ? land : lost;
      float gone = smoothstep(-0.7, -3.0, vAlive);                 // the copy's reply no longer begins a number: it does not answer
      base = mix(base, paper * 0.96, gone);
      float lit = max(dot(normalize(vN), light), 0.0);
      vec3 color = mix(base * vec3(0.8, 0.74, 0.68), base, 0.25 + 0.75 * lit);
      color = mix(color, ink, max(line, coast * 0.9) * (1.0 - gone));
      color = mix(color, ink, 0.6 * smoothstep(disc * 0.985, disc * 0.998, rr));
      color += (hash(gl_FragCoord.xy) - 0.5) * 0.025;
      gl_FragColor = vec4(color, 1.0);
    }`,
});
const geom = new THREE.PlaneGeometry(2 * R, 2 * R, UP - 1, UP - 1);
geom.setAttribute('alive', new THREE.BufferAttribute(new Float32Array(UP * UP), 1));   // best candidate's log p: low means the copy no longer answers
scene.add(new THREE.Mesh(geom, terrainMat));
// the sea: a translucent disc at margin zero, with a rim
const sea = new THREE.Mesh(new THREE.CircleGeometry(R * 0.999, 128), new THREE.MeshBasicMaterial({ color: hex('#dfe6ee'), transparent: true, opacity: 0.28, depthWrite: false, side: THREE.DoubleSide }));
scene.add(sea);
sea.visible = !SHOT;
const rim = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(Array.from({ length: 128 }, (_, k) => new THREE.Vector3(R * Math.cos(k / 128 * 2 * Math.PI), R * Math.sin(k / 128 * 2 * Math.PI), 0))),
                               new THREE.LineBasicMaterial({ color: hex(C.slate), transparent: true, opacity: 0.45 }));
function palette() {
  const u = terrainMat.uniforms;
  u.deep.value = hex(css('--land-deep')); u.warm.value = hex(css('--land-pale')); u.under.value = hex(css('--sea-pale')); u.abyss.value = hex(css('--sea-deep'));
  u.ramp.value = Number(css('--ramp')) || 0; sea.material.color = hex(css('--plane'));
  if (css('--contour')) u.ink.value = hex(css('--contour'));   // the contour lines, ink unless the page says otherwise
  if (Q.get('sea')) u.abyss.value = hex('#' + Q.get('sea'));   // stills: a paler deep blue
}
palette();
scene.add(rim);
rim.visible = !SHOT;
{ // a soft shadow
  const sc = document.createElement('canvas'); sc.width = sc.height = 256;
  const g = sc.getContext('2d'), grd = g.createRadialGradient(128, 128, 30, 128, 128, 128);
  grd.addColorStop(0, 'rgba(60, 40, 25, 0.22)'); grd.addColorStop(0.75, 'rgba(60, 40, 25, 0.04)'); grd.addColorStop(1, 'rgba(60, 40, 25, 0)');
  g.fillStyle = grd; g.fillRect(0, 0, 256, 256);
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(8.6, 8.6), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(sc), transparent: true, depthWrite: false }));
  shadow.position.z = -1.7; scene.add(shadow);
}
// the model: the centre of the slice
const model = new THREE.Mesh(new THREE.SphereGeometry(0.07, 24, 24), new THREE.MeshBasicMaterial({ color: hex(C.ink) }));
scene.add(model);
const stem = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineBasicMaterial({ color: hex(C.ink) }));
scene.add(stem);

// ------------------------------------------------------------------------------------------- time
function frameAt(x) { // x in [0, 1] of the reasoning → fractional frame index
  const fr = T.fr;
  if (x <= fr[0]) return 0;
  for (let i = 0; i < fr.length - 1; i++) if (x <= fr[i + 1]) return i + (x - fr[i]) / (fr[i + 1] - fr[i]);
  return fr.length - 1;
}
const ease = t => t * t * (3 - 2 * t);
function setTerrain(s) {
  const F = T.F, i = Math.min(Math.floor(s), F.length - 2), t = ease(s - i);
  const k = T.HS, a = T.fields[i], b = T.fields[i + 1];
  const pos = geom.attributes.position;
  for (let v = 0; v < pos.count; v++) {
    const row = Math.floor(v / UP), col = v % UP;
    const idx = col * UP + (UP - 1 - row);   // grid[a][b]: a along x, b along y; plane rows run top (y+) to bottom
    pos.setZ(v, k * (a[idx] + (b[idx] - a[idx]) * t));
  }
  pos.needsUpdate = true; geom.computeVertexNormals();
  const al = geom.attributes.alive, aa = T.alive[i], ab = T.alive[i + 1];
  for (let v = 0; v < al.count; v++) { const row = Math.floor(v / UP), col = v % UP, idx = col * UP + (UP - 1 - row); al.setX(v, aa[idx] + (ab[idx] - aa[idx]) * t); }
  al.needsUpdate = true;
  terrainMat.uniforms.hs.value = T.HS;
  controls.target.z = T.HS * T.mid * 0.9;   // keep the surface in frame wherever it sits
  const c = Math.floor(UP / 2), zc = k * (a[c * UP + c] + (b[c * UP + c] - a[c * UP + c]) * t);
  model.position.set(0, 0, zc + 0.02);
  stem.geometry.setFromPoints([new THREE.Vector3(0, 0, Math.min(0, zc)), new THREE.Vector3(0, 0, Math.max(0, zc) + 0.35)]);
}

// ------------------------------------------------------------------------------------------- text
const textEl = $('#text');
function charAt(x) { const s = frameAt(x), i = Math.min(Math.floor(s), T.F.length - 2), t = s - i; return Math.round(T.ends[i] + (T.ends[i + 1] - T.ends[i]) * t); }
const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
function setText(x) { // the reasoning, dark as far as the model has read, grey after
  const n = charAt(x), think = T.think;
  textEl.innerHTML = `<span class="read">${esc(think.slice(0, n))}</span><span class="mark"></span>${esc(think.slice(n))}`;
  textEl.scrollTop = Math.max(0, textEl.querySelector('.mark').offsetTop - textEl.offsetTop - 110);
}

// ------------------------------------------------------------------------------------------- loop
const params = new URLSearchParams(location.search);
let x = Number(params.get('t') || 0), playing = !params.has('t');
const slider = $('#t');
slider.oninput = () => { x = slider.value / 1000; playing = false; $('#play').textContent = 'play'; update(); };
$('#play').onclick = () => { playing = !playing; if (playing && x >= 1) x = 0; $('#play').textContent = playing ? 'pause' : 'play'; };
$('#play').textContent = playing ? 'pause' : 'play';
const picker = $('#traces');
const KIND = { resolving: 'reasoning resolves it', knew: 'knew it from the start', wrong: 'ends up wrong' };
TRACES.forEach((tr, i) => {
  const o = document.createElement('option');
  o.value = i; o.textContent = `${tr.q.length > 52 ? tr.q.slice(0, 50).trimEnd() + '…' : tr.q}  ·  ${KIND[tr.kind] || tr.kind}`; o.title = tr.q;
  picker.appendChild(o);
});
{ const k = Number(new URLSearchParams(location.search).get('trace') || 0); if (TRACES[k]) { T = TRACES[k]; picker.value = k; } }
const describe = () => {   // the question line and the legend's red entry, which follows a wrong answer in some traces
  $('#question').textContent = `${T.q} Right answer: ${T.right || T.a}.`;
  $('#legend-right').textContent = T.legend_right || 'right answer';
};
picker.onchange = () => { T = TRACES[Number(picker.value)]; describe(); x = 0; playing = true; $('#play').textContent = 'pause'; update(); };
describe();
function update() {
  const p = T.fr[0] + x * (T.fr[T.fr.length - 1] - T.fr[0]);   // the slider spans the measured stretch of the reasoning
  setTerrain(frameAt(p));
  setText(p);
  slider.value = Math.round(x * 1000);
}
function resize() { const w = box.clientWidth, h = box.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(box); resize();
update();
let last = performance.now();
function tick(now) {
  requestAnimationFrame(tick);
  const dt = (now - last) / 1000; last = now;
  if (playing) { x = Math.min(1, x + dt / 24); if (x >= 1) { playing = false; $('#play').textContent = 'play'; } update(); }
  if (sway) placeCamera(now);
  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(tick);
