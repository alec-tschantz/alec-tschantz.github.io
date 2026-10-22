// Animated backdrop for the home page: topographic lines over a slowly drifting landscape.
// Colours come from the --bg and --shade-1..4 CSS variables, faintest first.

const VERTEX = /* glsl */ `#version 300 es
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAGMENT = /* glsl */ `#version 300 es
precision highp float;

uniform vec2 u_res;
uniform float u_unit;
uniform float u_time;
uniform vec3 u_shades[5];
out vec4 fragColor;

// Background through the four shades, faintest to strongest.
vec3 palette(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 c = mix(u_shades[0], u_shades[1], smoothstep(0.00, 0.30, t));
  c = mix(c, u_shades[2], smoothstep(0.25, 0.60, t));
  c = mix(c, u_shades[3], smoothstep(0.55, 0.85, t));
  return mix(c, u_shades[4], smoothstep(0.85, 1.00, t));
}

float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

// 2D simplex noise, Ashima Arts / Stefan Gustavson (MIT).
vec3 permute(vec3 x) { return mod((x * 34.0 + 1.0) * x, 289.0); }

float snoise(vec2 v) {
  const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
  vec2 i = floor(v + dot(v, C.yy));
  vec2 x0 = v - i + dot(i, C.xx);
  vec2 i1 = x0.x > x0.y ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
  vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
  m = m * m * m * m;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 a0 = x - floor(x + 0.5);
  m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
  vec3 g = vec3(a0.x * x0.x + h.x * x0.y, a0.yz * x12.xz + h.yz * x12.yw);
  return 130.0 * dot(m, g);
}

float fbm(vec2 p) {
  float sum = 0.0;
  float amp = 0.5;
  for (int i = 0; i < 5; i++) {
    sum += amp * snoise(p);
    p = mat2(1.6, 1.2, -1.2, 1.6) * p;
    amp *= 0.5;
  }
  return sum;
}

void main() {
  float t = 3.1 * u_time;

  // Scaled by u_unit (the width, but never under 1200 CSS pixels), so the landscape
  // keeps its size however tall the canvas is, and phones see a calmer crop of it.
  vec2 q = (gl_FragCoord.xy - 0.5 * u_res) / u_unit;
  float vignette = smoothstep(1.0, 0.15, length(q));
  float h = fbm(4.05 * q + vec2(t * 0.012, -t * 0.008));

  // Ten lines per unit of height, every fourth stronger, slowly migrating uphill.
  float k = h * 10.0 - t * 0.12;
  float w = fwidth(k);
  float d = abs(fract(k + 0.5) - 0.5) / w;
  float major = mod(floor(k + 0.5), 4.0) < 0.5 ? 1.0 : 0.45;
  float line = (1.0 - smoothstep(0.5, 1.5, d)) * major * (1.0 - smoothstep(0.2, 0.5, w));

  float lift = smoothstep(-0.6, 0.8, h);
  vec3 color = mix(palette(0.18 * lift * vignette), palette(0.55 + 0.4 * lift), line * vignette);
  color += (hash(gl_FragCoord.xy + fract(t) * 1000.0) - 0.5) * 0.035;
  fragColor = vec4(color, 1.0);
}`;

function readShades() {
  const style = getComputedStyle(document.documentElement);
  return ["--bg", "--shade-1", "--shade-2", "--shade-3", "--shade-4"].flatMap((name) => {
    const hex = style.getPropertyValue(name).trim();
    return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  });
}

function start(canvas) {
  const gl = canvas.getContext("webgl2", { alpha: false, antialias: false });
  if (!gl) return;

  const program = gl.createProgram();
  for (const [type, source] of [[gl.VERTEX_SHADER, VERTEX], [gl.FRAGMENT_SHADER, FRAGMENT]]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    gl.attachShader(program, shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));

  gl.useProgram(program);
  gl.uniform3fv(gl.getUniformLocation(program, "u_shades"), readShades());
  const res = gl.getUniformLocation(program, "u_res");
  const unit = gl.getUniformLocation(program, "u_unit");
  const time = gl.getUniformLocation(program, "u_time");

  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let frame = 0;

  function draw(now) {
    const dpr = Math.min(devicePixelRatio, 2);
    const width = Math.round(canvas.clientWidth * dpr);
    const height = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      gl.viewport(0, 0, width, height);
    }
    gl.uniform2f(res, width, height);
    gl.uniform1f(unit, Math.max(width, 1200 * dpr));
    gl.uniform1f(time, reducedMotion ? 0 : now / 1000);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    frame = reducedMotion ? 0 : requestAnimationFrame(draw);
  }

  // With reduced motion only one frame is drawn, so redraw when the size changes.
  addEventListener("resize", () => { frame ||= requestAnimationFrame(draw); });
  frame = requestAnimationFrame(draw);
}

start(document.querySelector(".backdrop"));
