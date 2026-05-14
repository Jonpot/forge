export interface CommentColorOption {
  label: string;
  value: string;
}

export interface RgbColor {
  r: number;
  g: number;
  b: number;
}

export interface HsvColor {
  h: number;
  s: number;
  v: number;
}

export const DEFAULT_COMMENT_COLOR = "#64748b";

export const COMMENT_COLOR_OPTIONS: CommentColorOption[] = [
  { label: "Slate", value: DEFAULT_COMMENT_COLOR },
  { label: "Indigo", value: "#6366f1" },
  { label: "Blue", value: "#3b82f6" },
  { label: "Cyan", value: "#06b6d4" },
  { label: "Teal", value: "#14b8a6" },
  { label: "Emerald", value: "#22c55e" },
  { label: "Amber", value: "#f59e0b" },
  { label: "Orange", value: "#f97316" },
  { label: "Rose", value: "#f43f5e" },
];

const PALETTE_COLORS = new Set(COMMENT_COLOR_OPTIONS.map((option) => option.value));

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function expandShortHex(value: string): string {
  const [, raw] = value.match(/^#([0-9a-f]{3})$/i) ?? [];
  if (!raw) return value;
  return `#${raw
    .split("")
    .map((char) => `${char}${char}`)
    .join("")}`;
}

export function normalizeCommentColor(value: string | null | undefined): string | null {
  const text = (value ?? "").trim();
  if (!text) return null;
  const expanded = expandShortHex(text);
  if (!/^#[0-9a-f]{6}$/i.test(expanded)) {
    return null;
  }
  return expanded.toLowerCase();
}

export function resolveCommentColor(value: string | null | undefined): string {
  return normalizeCommentColor(value) ?? DEFAULT_COMMENT_COLOR;
}

export function isCommentPaletteColor(value: string | null | undefined): boolean {
  const normalized = normalizeCommentColor(value);
  return normalized !== null && PALETTE_COLORS.has(normalized);
}

export function hexToRgb(value: string): RgbColor {
  const normalized = resolveCommentColor(value);
  return {
    r: Number.parseInt(normalized.slice(1, 3), 16),
    g: Number.parseInt(normalized.slice(3, 5), 16),
    b: Number.parseInt(normalized.slice(5, 7), 16),
  };
}

export function rgbToHex(color: RgbColor): string {
  const toHex = (value: number) =>
    clamp(Math.round(value), 0, 255).toString(16).padStart(2, "0");
  return `#${toHex(color.r)}${toHex(color.g)}${toHex(color.b)}`;
}

export function rgbToHsv(color: RgbColor): HsvColor {
  const r = clamp(color.r / 255, 0, 1);
  const g = clamp(color.g / 255, 0, 1);
  const b = clamp(color.b / 255, 0, 1);
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;

  let h = 0;
  if (delta > 0) {
    if (max === r) {
      h = ((g - b) / delta) % 6;
    } else if (max === g) {
      h = (b - r) / delta + 2;
    } else {
      h = (r - g) / delta + 4;
    }
    h *= 60;
    if (h < 0) h += 360;
  }

  const s = max === 0 ? 0 : delta / max;
  return { h, s, v: max };
}

export function hsvToRgb(color: HsvColor): RgbColor {
  const h = ((color.h % 360) + 360) % 360;
  const s = clamp(color.s, 0, 1);
  const v = clamp(color.v, 0, 1);
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;

  let rPrime = 0;
  let gPrime = 0;
  let bPrime = 0;

  if (h < 60) {
    rPrime = c;
    gPrime = x;
  } else if (h < 120) {
    rPrime = x;
    gPrime = c;
  } else if (h < 180) {
    gPrime = c;
    bPrime = x;
  } else if (h < 240) {
    gPrime = x;
    bPrime = c;
  } else if (h < 300) {
    rPrime = x;
    bPrime = c;
  } else {
    rPrime = c;
    bPrime = x;
  }

  return {
    r: Math.round((rPrime + m) * 255),
    g: Math.round((gPrime + m) * 255),
    b: Math.round((bPrime + m) * 255),
  };
}

export function hsvToHex(color: HsvColor): string {
  return rgbToHex(hsvToRgb(color));
}

function rgba(value: string, alpha: number): string {
  const { r, g, b } = hexToRgb(value);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function buildCommentSwatchBackground(value: string): string {
  return resolveCommentColor(value);
}

export function buildCommentTheme(value: string | null | undefined, selected = false) {
  const color = resolveCommentColor(value);
  return {
    color,
    background: rgba(color, selected ? 0.14 : 0.09),
    border: rgba(color, selected ? 0.72 : 0.44),
    separator: rgba(color, selected ? 0.42 : 0.28),
    buttonFill: rgba(color, selected ? 0.18 : 0.12),
    buttonRing: rgba(color, selected ? 0.52 : 0.36),
    shadow: selected
      ? `0 0 0 1px ${rgba(color, 0.25)}, 0 4px 16px rgba(0, 0, 0, 0.4)`
      : `0 0 0 1px ${rgba(color, 0.15)}, 0 4px 16px rgba(0, 0, 0, 0.24)`,
    minimap: color,
    resizerLine: rgba(color, 0.82),
  };
}
