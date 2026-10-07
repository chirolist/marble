import { MAX_DROPS, MAX_RIPPLES, MAX_SPARKS, MAX_SPLATS, MAX_VORTICES } from "./tuning";

export const VERT = `#version 300 es
layout(location = 0) in vec2 aPosition;
out vec2 vUv;
void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

const HEADER = `#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 vUv;
out vec4 outColor;
`;

function frag(body: string): string {
  return `${HEADER}\n${body}`;
}

const BILINEAR = `
vec4 sampleLinear(sampler2D tex, vec2 uv, vec2 texel) {
  vec2 safe = clamp(uv, texel * 0.5, 1.0 - texel * 0.5);
  vec2 coord = safe / texel - 0.5;
  vec2 base = floor(coord);
  vec2 f = fract(coord);
  vec2 origin = (base + 0.5) * texel;
  vec4 a = texture(tex, origin);
  vec4 b = texture(tex, origin + vec2(texel.x, 0.0));
  vec4 c = texture(tex, origin + vec2(0.0, texel.y));
  vec4 d = texture(tex, origin + vec2(texel.x, texel.y));
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
`;

/** Keep idle-wave constants in sync with FluidSimulation.idleWave. */
const SURFACE = `
uniform sampler2D uWave;
uniform float uSurface;
uniform float uTime;
uniform float uWaveScale;
uniform float uMeniscus;

float waveHeight(float x) {
  float w = texture(uWave, vec2(clamp(x, 0.0, 1.0), 0.5)).r;
  return (w - 0.5) * uWaveScale;
}

float meniscusOf(float x) {
  float edge = abs(x - 0.5) * 2.0;
  return pow(edge, 3.0) * uMeniscus;
}

float idleWave(float x) {
  return sin(x * 18.0 + uTime * 1.1) * 0.0016
       + sin(x * 31.0 - uTime * 0.7) * 0.0008;
}

float surfaceAt(float x) {
  return uSurface + waveHeight(x) + meniscusOf(x) + idleWave(x);
}
`;

export const ADVECT_FRAG = frag(`
uniform sampler2D uVelocity;
uniform sampler2D uSource;
uniform vec2 uTexel;
uniform float uDt;
uniform float uDissipation;
uniform int uClampWater;
${SURFACE}
${BILINEAR}

void main() {
  vec2 vel = texture(uVelocity, vUv).xy;
  vec2 coord = vUv - uDt * vel * uTexel;
  if (uClampWater == 1) {
    float limit = surfaceAt(coord.x) - uTexel.y * 2.0;
    coord.y = min(coord.y, limit);
  }
  coord = clamp(coord, uTexel, vec2(1.0) - uTexel);
  vec4 result = sampleLinear(uSource, coord, uTexel);
  result.rgb *= uDissipation;
  if (uClampWater == 1) {
    result.rgb = max(result.rgb, vec3(0.0));
  }
  outColor = vec4(result.rgb, 1.0);
}
`);

export const CURL_FRAG = frag(`
uniform sampler2D uVelocity;
uniform vec2 uTexel;

void main() {
  vec2 l = vUv - vec2(uTexel.x, 0.0);
  vec2 r = vUv + vec2(uTexel.x, 0.0);
  vec2 b = vUv - vec2(0.0, uTexel.y);
  vec2 t = vUv + vec2(0.0, uTexel.y);
  float vyL = texture(uVelocity, l).y;
  float vyR = texture(uVelocity, r).y;
  float vxB = texture(uVelocity, b).x;
  float vxT = texture(uVelocity, t).x;
  float curl = vyR - vyL - (vxT - vxB);
  outColor = vec4(0.5 * curl, 0.0, 0.0, 1.0);
}
`);

export const FORCES_FRAG = frag(`
uniform sampler2D uVelocity;
uniform sampler2D uDye;
uniform sampler2D uCurl;
uniform vec2 uTexel;
uniform float uDt;
uniform float uBuoyancy;
uniform float uConfinement;
uniform float uAmbient;
${SURFACE}

vec2 ambientCurl(vec2 uv, float time) {
  float ax = 6.2831853 * 1.7;
  float ay = 6.2831853 * 2.15;
  float a = uv.x * ax + time * 0.16;
  float b = uv.y * ay - time * 0.12;
  vec2 c = vec2(sin(a) * cos(b) * ay, -cos(a) * ax * sin(b));

  float cx = 6.2831853 * 3.4;
  float cy = 6.2831853 * 2.7;
  float d = uv.x * cx - time * 0.1 + 1.7;
  float e = uv.y * cy + time * 0.08;
  c += 0.42 * vec2(sin(d) * cos(e) * cy, -cos(d) * cx * sin(e));
  return c;
}

void main() {
  float surface = surfaceAt(vUv.x);
  if (vUv.x < uTexel.x || vUv.x > 1.0 - uTexel.x || vUv.y < uTexel.y || vUv.y > surface) {
    outColor = vec4(0.0);
    return;
  }

  vec2 vel = texture(uVelocity, vUv).xy;
  vec3 dye = texture(uDye, vUv).rgb;

  float cL = texture(uCurl, vUv - vec2(uTexel.x, 0.0)).x;
  float cR = texture(uCurl, vUv + vec2(uTexel.x, 0.0)).x;
  float cB = texture(uCurl, vUv - vec2(0.0, uTexel.y)).x;
  float cT = texture(uCurl, vUv + vec2(0.0, uTexel.y)).x;
  float curl = texture(uCurl, vUv).x;

  vec2 grad = vec2(abs(cT) - abs(cB), abs(cR) - abs(cL));
  float glen = length(grad) + 0.0002;
  grad /= glen;
  vec2 confine = grad * curl * uConfinement;
  confine.y *= -1.0;
  vel += confine * uDt;

  float dens = dot(dye, vec3(0.2, 0.34, 0.18));
  float sink = dens / (1.0 + dens * 0.45);
  vel.y -= uBuoyancy * sink * uDt;

  float mask = smoothstep(0.015, 0.45, dens);
  vec2 flow = ambientCurl(vUv, uTime);
  vel += flow * uAmbient * mask * uDt;

  vel = clamp(vel, vec2(-52.0), vec2(52.0));
  outColor = vec4(vel, 0.0, 1.0);
}
`);

export const VORTEX_FRAG = frag(`
uniform sampler2D uVelocity;
uniform vec2 uTexel;
uniform int uCount;
uniform vec4 uV0[${MAX_VORTICES}];
uniform vec4 uV1[${MAX_VORTICES}];

void main() {
  vec2 pix = vUv / uTexel;
  vec2 vel = texture(uVelocity, vUv).xy;
  for (int i = 0; i < ${MAX_VORTICES}; i++) {
    if (i >= uCount) break;
    vec2 center = uV0[i].xy / uTexel;
    float sep = uV0[i].z;
    float radius = max(uV0[i].w, 1.0);
    float strength = uV1[i].x;
    float jet = uV1[i].y;
    float spin = uV1[i].z;
    float spread = uV1[i].w;

    vec2 pL = pix - (center + vec2(-sep, 0.0));
    vec2 pR = pix - (center + vec2(sep, 0.0));
    float gL = exp(-dot(pL, pL) / (radius * radius));
    float gR = exp(-dot(pR, pR) / (radius * radius));
    vec2 vL = vec2(pL.y, -pL.x) / radius * gL;
    vec2 vR = vec2(-pR.y, pR.x) / radius * gR;
    vel += (vL + vR) * strength * spin;

    vec2 p = pix - center;
    float g = exp(-dot(p, p) / (radius * radius * 1.6));
    vel.y -= jet * g;
    vel.x += (p.x / radius) * g * spread;
  }
  outColor = vec4(vel, 0.0, 1.0);
}
`);

export const DIVERGENCE_FRAG = frag(`
uniform sampler2D uVelocity;
uniform vec2 uTexel;
${SURFACE}

void main() {
  vec2 l = vUv - vec2(uTexel.x, 0.0);
  vec2 r = vUv + vec2(uTexel.x, 0.0);
  vec2 b = vUv - vec2(0.0, uTexel.y);
  vec2 t = vUv + vec2(0.0, uTexel.y);
  vec2 vL = texture(uVelocity, l).xy;
  vec2 vR = texture(uVelocity, r).xy;
  vec2 vB = texture(uVelocity, b).xy;
  vec2 vT = texture(uVelocity, t).xy;
  if (vUv.y > surfaceAt(vUv.x)) {
    outColor = vec4(0.0);
    return;
  }
  float div = 0.5 * (vR.x - vL.x + vT.y - vB.y);
  outColor = vec4(div, 0.0, 0.0, 1.0);
}
`);

export const PRESSURE_FRAG = frag(`
uniform sampler2D uPressure;
uniform sampler2D uDivergence;
uniform vec2 uTexel;

void main() {
  float L = texture(uPressure, vUv - vec2(uTexel.x, 0.0)).x;
  float R = texture(uPressure, vUv + vec2(uTexel.x, 0.0)).x;
  float B = texture(uPressure, vUv - vec2(0.0, uTexel.y)).x;
  float T = texture(uPressure, vUv + vec2(0.0, uTexel.y)).x;
  float div = texture(uDivergence, vUv).x;
  float pressure = (L + R + B + T - div) * 0.25;
  outColor = vec4(pressure, 0.0, 0.0, 1.0);
}
`);

export const GRADIENT_FRAG = frag(`
uniform sampler2D uPressure;
uniform sampler2D uVelocity;
uniform vec2 uTexel;
${SURFACE}

void main() {
  float surface = surfaceAt(vUv.x);
  if (vUv.x < uTexel.x || vUv.x > 1.0 - uTexel.x || vUv.y < uTexel.y || vUv.y > surface) {
    outColor = vec4(0.0);
    return;
  }
  float L = texture(uPressure, vUv - vec2(uTexel.x, 0.0)).x;
  float R = texture(uPressure, vUv + vec2(uTexel.x, 0.0)).x;
  float B = texture(uPressure, vUv - vec2(0.0, uTexel.y)).x;
  float T = texture(uPressure, vUv + vec2(0.0, uTexel.y)).x;
  vec2 vel = texture(uVelocity, vUv).xy;
  vel -= 0.5 * vec2(R - L, T - B);
  vel = clamp(vel, vec2(-52.0), vec2(52.0));
  outColor = vec4(vel, 0.0, 1.0);
}
`);

export const DIFFUSE_FRAG = frag(`
uniform sampler2D uDye;
uniform vec2 uTexel;
uniform float uDt;
uniform float uDiffuse;
${SURFACE}

vec3 sampleDye(vec2 uv, vec3 fallback) {
  if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= surfaceAt(uv.x)) {
    return fallback;
  }
  return texture(uDye, uv).rgb;
}

void main() {
  vec3 c = texture(uDye, vUv).rgb;
  if (vUv.y > surfaceAt(vUv.x)) {
    outColor = vec4(0.0);
    return;
  }
  vec3 avg = (
    sampleDye(vUv - vec2(uTexel.x, 0.0), c) +
    sampleDye(vUv + vec2(uTexel.x, 0.0), c) +
    sampleDye(vUv - vec2(0.0, uTexel.y), c) +
    sampleDye(vUv + vec2(0.0, uTexel.y), c)
  ) * 0.25;
  float k = 1.0 - exp(-uDiffuse * uDt);
  outColor = vec4(max(mix(c, avg, k), vec3(0.0)), 1.0);
}
`);

export const SPLAT_FRAG = frag(`
uniform sampler2D uTarget;
uniform vec2 uTexel;
uniform int uCount;
uniform vec4 uS0[${MAX_SPLATS}];
uniform vec4 uS1[${MAX_SPLATS}];
uniform vec3 uC0[${MAX_SPLATS}];

void main() {
  vec3 acc = texture(uTarget, vUv).rgb;
  vec2 pix = vUv / uTexel;
  for (int i = 0; i < ${MAX_SPLATS}; i++) {
    if (i >= uCount) break;
    vec2 center = uS0[i].xy / uTexel;
    float radius = max(uS0[i].z, 0.5);
    float scale = uS0[i].w;
    vec2 stretch = max(uS1[i].xy, vec2(0.05));
    vec2 p = (pix - center) / stretch;
    float g = exp(-dot(p, p) / (radius * radius));
    acc += uC0[i] * scale * g;
  }
  outColor = vec4(min(max(acc, vec3(0.0)), vec3(6.0)), 1.0);
}
`);

export const MASK_FRAG = frag(`
uniform sampler2D uDye;
uniform vec2 uTexel;
${SURFACE}

void main() {
  if (vUv.y > surfaceAt(vUv.x) + 0.012) {
    outColor = vec4(0.0);
    return;
  }
  outColor = vec4(texture(uDye, vUv).rgb, 1.0);
}
`);

export const DISPLAY_FRAG = frag(`
uniform sampler2D uDye;
uniform vec2 uTexel;
uniform vec2 uCanvas;
${SURFACE}
${BILINEAR}

uniform int uDropCount;
uniform vec4 uDrops[${MAX_DROPS}];
uniform vec3 uDropColor[${MAX_DROPS}];
uniform vec4 uDropMeta[${MAX_DROPS}];

uniform int uSparkCount;
uniform vec4 uSparks[${MAX_SPARKS}];
uniform vec3 uSparkColor[${MAX_SPARKS}];

uniform int uRippleCount;
uniform vec4 uRipples[${MAX_RIPPLES}];

void main() {
  float surface = surfaceAt(vUv.x);
  float slope = surfaceAt(vUv.x + uTexel.x * 2.0) - surfaceAt(vUv.x - uTexel.x * 2.0);
  float below = 1.0 - smoothstep(surface - 0.001, surface + 0.004, vUv.y);

  vec2 sampleUv = vUv;
  float nearSurface = exp(-max(surface - vUv.y, 0.0) * 16.0);
  sampleUv.x -= slope * 7.0 * nearSurface;
  float side = min(vUv.x, 1.0 - vUv.x);
  float pinch = smoothstep(0.14, 0.0, side);
  sampleUv.x -= pinch * 0.012 * sign(vUv.x - 0.5);
  sampleUv = clamp(sampleUv, uTexel, vec2(1.0) - uTexel);

  vec3 dye = sampleLinear(uDye, sampleUv, uTexel).rgb;
  vec3 blur = (
    sampleLinear(uDye, sampleUv + vec2(uTexel.x, 0.0), uTexel).rgb +
    sampleLinear(uDye, sampleUv - vec2(uTexel.x, 0.0), uTexel).rgb +
    sampleLinear(uDye, sampleUv + vec2(0.0, uTexel.y), uTexel).rgb +
    sampleLinear(uDye, sampleUv - vec2(0.0, uTexel.y), uTexel).rgb
  ) * 0.25;
  dye = max(dye + (dye - blur) * 0.42, vec3(0.0));

  vec3 waterDeep = vec3(0.74, 0.85, 0.89);
  vec3 waterShallow = vec3(0.89, 0.94, 0.95);
  float depthT = smoothstep(0.0, surface, vUv.y);
  vec3 water = mix(waterDeep, waterShallow, depthT);
  water += vec3(0.05, 0.055, 0.05) * smoothstep(surface - 0.28, surface, vUv.y);

  float n1 = sin(vUv.x * 18.0 + uTime * 0.22 + sin(vUv.y * 9.0 - uTime * 0.12));
  float n2 = sin(vUv.y * 14.0 - uTime * 0.16 + vUv.x * 7.0);
  float caustic = smoothstep(0.72, 1.0, n1 * n2);

  vec3 raw = exp(-max(dye, vec3(0.0)));
  float luma = dot(raw, vec3(0.299, 0.587, 0.114));
  vec3 vivid = clamp(mix(vec3(luma), raw, 3.8), 0.0, 1.0);
  float pk = max(vivid.r, max(vivid.g, vivid.b));
  vivid /= max(pk, 1e-3);
  float thick = dot(dye, vec3(1.05));
  float shade = exp(-thick * 0.32);
  vec3 ink = vivid * mix(0.5, 1.0, shade);
  float cover = 1.0 - exp(-thick * 4.4);
  vec3 col = mix(water, ink, cover);
  col += vec3(0.035, 0.04, 0.038) * caustic * (1.0 - cover) * below;

  vec3 air = mix(vec3(0.80, 0.86, 0.89), vec3(0.93, 0.95, 0.96), vUv.y);
  float airShade = smoothstep(0.0, 0.18, min(vUv.x, 1.0 - vUv.x));
  air *= mix(0.9, 1.0, airShade);
  col = mix(col, air, 1.0 - below);

  float dSurf = vUv.y - surface;
  float line = smoothstep(0.0042, 0.0, abs(dSurf));
  float crest = clamp(slope * 50.0, 0.0, 1.0);
  vec3 spec = vec3(1.0, 0.985, 0.96);
  col = mix(col, spec, line * (0.55 + crest * 0.35));
  float under = smoothstep(0.0, -0.006, dSurf) * smoothstep(-0.03, -0.008, dSurf);
  col *= mix(1.0, 0.9, under * below);

  vec2 pixel = vUv * uCanvas;
  for (int i = 0; i < ${MAX_RIPPLES}; i++) {
    if (i >= uRippleCount) break;
    float age = uRipples[i].y;
    float strength = uRipples[i].z;
    float radius = age * 0.16;
    vec2 d = vec2(vUv.x - uRipples[i].x, (vUv.y - surface) * 3.4);
    float dist = length(d);
    float ring = exp(-abs(dist - radius) * 70.0);
    float fade = exp(-age * 1.55) * strength;
    float band = exp(-abs(vUv.y - surface) * 28.0);
    col += vec3(1.0, 0.99, 0.97) * ring * fade * band * 0.55;
  }

  float edgeShade = smoothstep(0.0, 0.055, side);
  col *= mix(0.8, 1.0, edgeShade);
  col *= mix(0.88, 1.0, smoothstep(0.0, 0.07, vUv.y));
  float rim = smoothstep(0.018, 0.0, side) * smoothstep(0.0, 0.0035, side);
  col += vec3(1.0) * rim * 0.28 * mix(below, 1.0, 0.35);

  for (int i = 0; i < ${MAX_DROPS}; i++) {
    if (i >= uDropCount) break;
    vec2 center = uDrops[i].xy * uCanvas;
    float rad = uDrops[i].z;
    float stretch = max(uDrops[i].w, 0.55);
    float squash = clamp(uDropMeta[i].x, 0.0, 1.0);
    float wobble = uDropMeta[i].y;
    vec2 p = pixel - center;
    p.x -= wobble * uCanvas.x;

    float closeness = smoothstep(0.16, 0.0, uDrops[i].y - surface);
    vec2 sh = vec2(vUv.x - uDrops[i].x, (vUv.y - surface) * 2.4);
    float shadow = exp(-dot(sh, sh) * 90.0) * closeness * (1.0 - squash);
    col *= mix(1.0, 0.9, shadow * below);

    float sx = mix(1.0, 1.7, squash);
    float sy = mix(stretch, 0.4, squash);
    vec2 q = vec2(p.x / sx, p.y / sy);
    float body = smoothstep(rad, rad * 0.42, length(q));

    float tail = 0.0;
    float rise = p.y / max(rad, 0.001);
    if (rise > 0.05 && squash < 0.3) {
      float reach = stretch * 2.15;
      float halfW = rad * mix(0.62, 0.04, clamp(rise / reach, 0.0, 1.0));
      tail = smoothstep(halfW, halfW * 0.25, abs(p.x));
      tail *= smoothstep(reach, 0.15, rise);
      tail *= 0.85 * (1.0 - squash / 0.3);
    }

    float alpha = max(body, tail);
    alpha *= mix(1.0, 0.0, smoothstep(0.28, 0.92, squash));
    if (alpha > 0.002) {
      vec3 pigmentColor = uDropColor[i];
      vec2 specP = vec2(p.x / sx + rad * 0.32, p.y / sy - rad * 0.34);
      float highlight = smoothstep(rad * 0.48, rad * 0.04, length(specP)) * (1.0 - squash);
      vec3 dropCol = mix(pigmentColor * vec3(0.72), pigmentColor, smoothstep(0.0, 0.55, body));
      dropCol = mix(dropCol, vec3(1.0), highlight * 0.62);
      float rimDark = smoothstep(rad * 0.95, rad * 0.45, length(q));
      dropCol *= mix(0.78, 1.0, max(rimDark, tail));
      col = mix(col, dropCol, clamp(alpha, 0.0, 1.0));
    }
  }

  for (int i = 0; i < ${MAX_SPARKS}; i++) {
    if (i >= uSparkCount) break;
    vec2 sp = pixel - uSparks[i].xy * uCanvas;
    float rad = uSparks[i].z;
    float alpha = uSparks[i].w;
    float d = length(sp);
    float a = smoothstep(rad, rad * 0.2, d) * alpha;
    if (a > 0.002) {
      vec3 sc = mix(uSparkColor[i], vec3(1.0), smoothstep(rad, 0.0, d) * 0.35);
      col = mix(col, sc, clamp(a, 0.0, 0.9));
    }
  }

  float grain = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  col += (grain - 0.5) * 0.008;
  outColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`);
