export type RGB = [number, number, number];

export type Swatch = {
  id: string;
  name: string;
  hex: string;
  rgb: RGB;
};

/**
 * Artist pigments, not display-pure primaries.
 * Blue keeps some green reflectance so it mixes with yellow toward green.
 * Red keeps a little green so it mixes with yellow toward orange.
 */
export const SWATCHES: Swatch[] = [
  { id: "red", name: "赤", hex: "#e2472f", rgb: [0.886, 0.278, 0.184] },
  { id: "blue", name: "青", hex: "#1c4fd4", rgb: [0.11, 0.31, 0.831] },
  { id: "yellow", name: "黄", hex: "#f2c41a", rgb: [0.949, 0.769, 0.102] },
  { id: "green", name: "緑", hex: "#1c9a52", rgb: [0.11, 0.604, 0.322] },
  { id: "purple", name: "紫", hex: "#7a3aad", rgb: [0.478, 0.227, 0.678] },
  { id: "orange", name: "橙", hex: "#f07818", rgb: [0.941, 0.471, 0.094] },
];

export function hexToRgb(hex: string): RGB {
  const value = hex.replace("#", "").trim();
  const n = Number.parseInt(value, 16);
  if (!Number.isFinite(n) || value.length !== 6) {
    return [0.8, 0.2, 0.15];
  }
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

export function rgbToHex(rgb: RGB): string {
  const channel = (c: number) =>
    Math.round(Math.min(1, Math.max(0, c)) * 255)
      .toString(16)
      .padStart(2, "0");
  return `#${channel(rgb[0])}${channel(rgb[1])}${channel(rgb[2])}`;
}

/**
 * Convert a display color into Beer's-law absorption.
 * Overlapping ink adds absorption, so transmittance multiplies:
 * red and blue darken into purple, blue and yellow into green.
 */
export function pigmentAbsorption(rgb: RGB): RGB {
  let r = rgb[0];
  let g = rgb[1];
  const b = rgb[2];
  const max = Math.max(r, g, b);

  if (max < 0.04) {
    return [1.35, 1.35, 1.35];
  }

  if (b > r + 0.18 && b >= g && g < b * 0.36) {
    g = b * 0.36;
    r = Math.min(r, b * 0.14);
  }
  if (r > b + 0.25 && r > g + 0.2 && g < r * 0.28) {
    g = Math.max(g, r * 0.26);
  }

  const eps = 0.04;
  const scale = 1;
  return [
    -Math.log(Math.max(r, eps)) * scale,
    -Math.log(Math.max(g, eps)) * scale,
    -Math.log(Math.max(b, eps)) * scale,
  ];
}
