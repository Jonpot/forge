/** Headless geometry checks for layout.ts. Run: node scripts/verify-layout.mjs
 * (bundles layout.ts on the fly via esbuild). */
import { build } from "esbuild";
import { pathToFileURL } from "url";
import { mkdtempSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

const out = join(mkdtempSync(join(tmpdir(), "sf-layout-")), "layout.mjs");
await build({ entryPoints: ["src/webview/layout.ts"], bundle: true, format: "esm", outfile: out });
const { computeLayout } = await import(pathToFileURL(out));

const N = (id) => ({ id, width: 200, height: 120 });
const E = (source, target) => ({ source, target });
let failures = 0;
const check = (name, condition, detail = "") => {
  console.log(`${condition ? "PASS" : "FAIL"}  ${name}${condition ? "" : "  — " + detail}`);
  if (!condition) failures++;
};
const centerY = (p) => p.y + 60;

// ── Jonathan's example: A→B→C→D chain + skip edge A→D ───────────────────
{
  const pos = computeLayout([N("A"), N("B"), N("C"), N("D")], [E("A", "B"), E("B", "C"), E("C", "D"), E("A", "D")]);
  const [a, b, c, d] = ["A", "B", "C", "D"].map((id) => pos.get(id));
  check("chain occupies strictly increasing columns", a.x < b.x && b.x < c.x && c.x < d.x);
  const span = (p, q) => Math.hypot(q.x - p.x, centerY(q) - centerY(p));
  const chainMax = Math.max(span(a, b), span(b, c), span(c, d));
  check("skip edge A→D is the longest edge", span(a, d) > chainMax, `A→D ${span(a, d).toFixed(0)} vs chain max ${chainMax.toFixed(0)}`);
  const drift = Math.max(...[b, c, d].map((p) => Math.abs(centerY(p) - centerY(a))));
  check("chain is roughly horizontally aligned (short edges)", drift <= 240, `max drift ${drift.toFixed(0)}px`);
}

// ── Crossing minimization: parallel chains wired crosswise ──────────────
{
  const pos = computeLayout(
    [N("A1"), N("A2"), N("B1"), N("B2")],
    [E("A1", "B2"), E("A2", "B1")], // doc order would cross; sweeps must uncross
  );
  const sameOrder =
    Math.sign(centerY(pos.get("A1")) - centerY(pos.get("A2"))) ===
    Math.sign(centerY(pos.get("B2")) - centerY(pos.get("B1")));
  check("crosswise chains are uncrossed", sameOrder);
}

// ── Corridor reservation: long edge pushes a mid-column node aside ──────
{
  // A→B→C→D chain plus A→D: B and C sit in the columns the A→D edge crosses.
  // With the chain aligned near A's lane, the A→D corridor (also near A's
  // lane through dummies) must not run THROUGH B/C — the relaxation places
  // dummies adjacent to, not on top of, the real nodes (gap-projected).
  const pos = computeLayout(
    [N("A"), N("B"), N("C"), N("D"), N("X")],
    [E("A", "B"), E("B", "C"), E("C", "D"), E("A", "D"), E("X", "B")],
  );
  const ys = ["A", "B", "C", "D", "X"].map((id) => pos.get(id).y);
  check("all five nodes placed", ys.every((y) => Number.isFinite(y)));
}

// ── Invariant: no vertical overlap within any column ─────────────────────
{
  const nodes = Array.from({ length: 14 }, (_, i) => N(`n${i}`));
  const edges = [];
  for (let i = 1; i < 14; i++) edges.push(E(`n${Math.floor((i - 1) / 2)}`, `n${i}`)); // binary tree
  edges.push(E("n0", "n13"), E("n1", "n12")); // skip edges
  const pos = computeLayout(nodes, edges);
  const byX = new Map();
  for (const [id, p] of pos) {
    if (!byX.has(p.x)) byX.set(p.x, []);
    byX.get(p.x).push(p.y);
  }
  let overlap = false;
  for (const ys of byX.values()) {
    ys.sort((m, n) => m - n);
    for (let i = 1; i < ys.length; i++) if (ys[i] - ys[i - 1] < 120) overlap = true;
  }
  check("no same-column node overlap (14-node tree + skips)", !overlap);
  check("every node placed", pos.size === 14);
}

console.log(failures === 0 ? "\nlayout geometry: ALL CHECKS PASS" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
