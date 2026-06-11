/** Category icons + colors, ported from the desktop app's
 * frontend/src/utils/categoryStyles.ts so *Forge nodes read as Forge nodes.
 *
 * Desktop categories are curated names (IO, Transform, …); *Forge categories
 * default to module paths, so the deterministic hash fallback does most of
 * the work here — same algorithm as desktop, so a category that exists in
 * both worlds gets the same color. (User overrides come later, with settings.)
 */

export interface CategoryStyle {
  icon: string;
  accent: string; // ~tailwind 400 — icons, accents
  badgeText: string; // ~tailwind 300
  badgeBg: string; // ~tailwind 900 @ 50%
}

const DEFAULT_ICON: Record<string, string> = {
  IO: "⇄",
  Operator: "+",
  Combine: "⊕",
  Transform: "Δ",
  Statistics: "σ",
  Clustering: "⊙",
  Factorization: "⊗",
  Dimensionality: "ℝ",
  Visualization: "📈",
  Special: "★",
  Custom: "★",
  "Built-in": "★",
};

// key, 400, 300, 900 — order matches desktop CATEGORY_COLOR_OPTIONS so the
// hash fallback lands on the same color for the same category name.
const COLORS: [string, string, string, string][] = [
  ["violet", "#a78bfa", "#c4b5fd", "#4c1d95"],
  ["green", "#4ade80", "#86efac", "#14532d"],
  ["amber", "#fbbf24", "#fcd34d", "#78350f"],
  ["sky", "#38bdf8", "#7dd3fc", "#0c4a6e"],
  ["blue", "#60a5fa", "#93c5fd", "#1e3a8a"],
  ["emerald", "#34d399", "#6ee7b7", "#064e3b"],
  ["pink", "#f472b6", "#f9a8d4", "#831843"],
  ["orange", "#fb923c", "#fdba74", "#7c2d12"],
  ["teal", "#2dd4bf", "#5eead4", "#134e4a"],
  ["yellow", "#facc15", "#fde047", "#713f12"],
  ["purple", "#c084fc", "#d8b4fe", "#581c87"],
  ["rose", "#fb7185", "#fda4af", "#881337"],
  ["fuchsia", "#e879f9", "#f0abfc", "#701a75"],
  ["lime", "#a3e635", "#bef264", "#365314"],
  ["cyan", "#22d3ee", "#67e8f9", "#164e63"],
  ["indigo", "#818cf8", "#a5b4fc", "#312e81"],
  ["red", "#f87171", "#fca5a5", "#7f1d1d"],
];

const COLOR_BY_KEY = new Map(COLORS.map((c) => [c[0], c]));

const DEFAULT_COLOR_KEY: Record<string, string> = {
  IO: "violet",
  Operator: "green",
  Combine: "amber",
  Transform: "sky",
  Statistics: "blue",
  Clustering: "emerald",
  Visualization: "pink",
  Factorization: "orange",
  Dimensionality: "teal",
  Special: "yellow",
  Custom: "purple",
  "Built-in": "yellow",
};

function categoryHash(category: string): number {
  let hash = 0;
  for (let i = 0; i < category.length; i += 1) {
    hash = (hash * 31 + category.charCodeAt(i)) >>> 0;
  }
  return hash;
}

export function resolveCategoryStyle(category: string): CategoryStyle {
  const key = DEFAULT_COLOR_KEY[category] ?? COLORS[categoryHash(category) % COLORS.length][0];
  const [, accent, badgeText, badgeBg] = COLOR_BY_KEY.get(key) ?? COLORS[0];
  return {
    icon: DEFAULT_ICON[category] ?? "◈",
    accent,
    badgeText,
    badgeBg: badgeBg + "80", // 50% alpha, matching desktop's bg-*-900/50
  };
}
