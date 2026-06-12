/** Layered (Sugiyama-style) pipeline layout.
 *
 * Phases:
 *   1. Longest-path layering — a node sits one column right of its deepest
 *      parent. This alone gives the "cluster short, skip long" property:
 *      in A→B→C→D with a direct A→D, the chain edges are one column each
 *      while A→D spans three.
 *   2. Virtual nodes — an edge spanning k>1 columns drops an invisible
 *      placeholder into each intermediate column. They participate in
 *      ordering and spacing, so long edges reserve a corridor and real nodes
 *      are pushed out of their path (nodes render above edges; an edge
 *      passing under an unrelated node reads as a connection).
 *   3. Crossing minimization — alternating forward/backward barycenter
 *      sweeps: each column is reordered by the mean position of its
 *      neighbors in the adjacent column.
 *   4. Coordinates — per-column stacking with measured sizes, then
 *      relaxation passes that pull every item toward the mean of its
 *      neighbors' centers (connected nodes align; edges shorten and
 *      straighten) while a projection pass preserves the crossing-free
 *      order and minimum gaps.
 *
 * Pure module: no React/ReactFlow imports, unit-verifiable headlessly.
 */

export interface LayoutNode {
  id: string;
  width: number;
  height: number;
}

export interface LayoutEdge {
  source: string;
  target: string;
}

export interface LayoutOptions {
  columnGap: number;
  rowGap: number;
  marginX: number;
  marginY: number;
  sweeps: number;
  relaxRounds: number;
  dummyHeight: number;
}

const DEFAULTS: LayoutOptions = {
  columnGap: 140,
  rowGap: 48,
  marginX: 80,
  marginY: 60,
  sweeps: 4,
  relaxRounds: 10,
  dummyHeight: 24,
};

interface Item {
  id: string;
  real: boolean;
  layer: number;
  height: number;
  width: number;
  y: number; // top
  up: Item[];
  down: Item[];
}

export function computeLayout(
  nodes: LayoutNode[],
  edges: LayoutEdge[],
  options: Partial<LayoutOptions> = {},
): Map<string, { x: number; y: number }> {
  const opt = { ...DEFAULTS, ...options };
  const result = new Map<string, { x: number; y: number }>();
  if (nodes.length === 0) return result;

  const byId = new Map(nodes.map((n) => [n.id, n]));
  const realEdges = edges.filter(
    (e) => byId.has(e.source) && byId.has(e.target) && e.source !== e.target,
  );

  // ---- 1. longest-path layering (Kahn) --------------------------------
  const indegree = new Map(nodes.map((n) => [n.id, 0]));
  const children = new Map<string, string[]>(nodes.map((n) => [n.id, []]));
  for (const e of realEdges) {
    indegree.set(e.target, (indegree.get(e.target) ?? 0) + 1);
    children.get(e.source)!.push(e.target);
  }
  const layerOf = new Map<string, number>();
  const queue = nodes.filter((n) => (indegree.get(n.id) ?? 0) === 0).map((n) => n.id);
  queue.forEach((id) => layerOf.set(id, 0));
  while (queue.length > 0) {
    const id = queue.shift()!;
    for (const child of children.get(id) ?? []) {
      layerOf.set(child, Math.max(layerOf.get(child) ?? 0, (layerOf.get(id) ?? 0) + 1));
      indegree.set(child, (indegree.get(child) ?? 1) - 1);
      if (indegree.get(child) === 0) queue.push(child);
    }
  }
  for (const n of nodes) if (!layerOf.has(n.id)) layerOf.set(n.id, 0); // cycle leftovers

  // ---- 2. items + virtual nodes ----------------------------------------
  const items = new Map<string, Item>();
  for (const n of nodes) {
    items.set(n.id, {
      id: n.id,
      real: true,
      layer: layerOf.get(n.id)!,
      height: Math.max(40, n.height),
      width: Math.max(120, n.width),
      y: 0,
      up: [],
      down: [],
    });
  }
  const link = (a: Item, b: Item) => {
    a.down.push(b);
    b.up.push(a);
  };
  let dummySeq = 0;
  for (const e of realEdges) {
    const source = items.get(e.source)!;
    const target = items.get(e.target)!;
    if (target.layer <= source.layer) continue; // cycle edge — skip routing
    let prev = source;
    for (let layer = source.layer + 1; layer < target.layer; layer++) {
      const dummy: Item = {
        id: `__dummy_${dummySeq++}`,
        real: false,
        layer,
        height: opt.dummyHeight,
        width: 0,
        y: 0,
        up: [],
        down: [],
      };
      items.set(dummy.id, dummy);
      link(prev, dummy);
      prev = dummy;
    }
    link(prev, target);
  }

  const maxLayer = Math.max(...[...items.values()].map((i) => i.layer));
  const layers: Item[][] = Array.from({ length: maxLayer + 1 }, () => []);
  for (const item of items.values()) layers[item.layer].push(item);

  // ---- 3. barycenter crossing-minimization sweeps -----------------------
  const indexIn = (layer: Item[]): Map<Item, number> => new Map(layer.map((it, i) => [it, i]));
  const sortByBarycenter = (layer: Item[], neighborsOf: (it: Item) => Item[], ref: Map<Item, number>) => {
    const current = indexIn(layer);
    const score = new Map<Item, number>();
    for (const it of layer) {
      const ns = neighborsOf(it);
      score.set(
        it,
        ns.length === 0
          ? current.get(it)! // keep position when unconstrained
          : ns.reduce((sum, n) => sum + (ref.get(n) ?? 0), 0) / ns.length,
      );
    }
    layer.sort((a, b) => score.get(a)! - score.get(b)! || current.get(a)! - current.get(b)!);
  };
  for (let sweep = 0; sweep < opt.sweeps; sweep++) {
    for (let l = 1; l <= maxLayer; l++) {
      sortByBarycenter(layers[l], (it) => it.up, indexIn(layers[l - 1]));
    }
    for (let l = maxLayer - 1; l >= 0; l--) {
      sortByBarycenter(layers[l], (it) => it.down, indexIn(layers[l + 1]));
    }
  }

  // ---- 4. coordinates ----------------------------------------------------
  const stack = (layer: Item[]) => {
    let y = opt.marginY;
    for (const it of layer) {
      it.y = y;
      y += it.height + opt.rowGap;
    }
  };
  layers.forEach(stack);

  const center = (it: Item) => it.y + it.height / 2;
  const projectOrder = (layer: Item[], desired: Map<Item, number>) => {
    // Forward pass enforces order + gaps from desired tops; backward pass
    // pulls slack upward; the average of both stays gap-feasible after one
    // final forward fix-up.
    const forward = new Map<Item, number>();
    let floor = -Infinity;
    for (const it of layer) {
      const y = Math.max(desired.get(it)!, floor);
      forward.set(it, y);
      floor = y + it.height + opt.rowGap;
    }
    const backward = new Map<Item, number>();
    let ceiling = Infinity;
    for (let i = layer.length - 1; i >= 0; i--) {
      const it = layer[i];
      const y = Math.min(desired.get(it)!, ceiling - it.height);
      backward.set(it, y);
      ceiling = y - opt.rowGap;
    }
    floor = -Infinity;
    for (const it of layer) {
      const y = Math.max((forward.get(it)! + backward.get(it)!) / 2, floor);
      it.y = y;
      floor = y + it.height + opt.rowGap;
    }
  };
  for (let round = 0; round < opt.relaxRounds; round++) {
    const sweepLayers = round % 2 === 0 ? layers : [...layers].reverse();
    for (const layer of sweepLayers) {
      const desired = new Map<Item, number>();
      for (const it of layer) {
        const ns = [...it.up, ...it.down];
        desired.set(
          it,
          ns.length === 0 ? it.y : ns.reduce((s, n) => s + center(n), 0) / ns.length - it.height / 2,
        );
      }
      projectOrder(layer, desired);
    }
  }

  // Normalize: top of the diagram at marginY.
  const minY = Math.min(...[...items.values()].map((it) => it.y));
  const shift = opt.marginY - minY;

  let x = opt.marginX;
  for (const layer of layers) {
    const columnWidth = Math.max(...layer.map((it) => it.width), 180);
    for (const it of layer) {
      if (it.real) result.set(it.id, { x: Math.round(x), y: Math.round(it.y + shift) });
    }
    x += columnWidth + opt.columnGap;
  }
  return result;
}
