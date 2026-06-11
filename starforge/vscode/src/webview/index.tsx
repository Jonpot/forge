import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactFlow, {
  Background,
  BackgroundVariant,
  Connection,
  Controls,
  Edge,
  Handle,
  MiniMap,
  Node,
  NodeProps,
  NodeResizer,
  Position,
  ReactFlowInstance,
  ReactFlowProvider,
  SelectionMode,
  useEdgesState,
  useNodesState,
} from "reactflow";
import "reactflow/dist/style.css";
import "./styles.css";
import { resolveCategoryStyle } from "./categoryStyles";

declare function acquireVsCodeApi(): { postMessage(message: any): void };
const vscode = acquireVsCodeApi();

// ---------------------------------------------------------------- doc model

interface ParamInfo {
  name: string;
  annotation: string | null;
  default_repr: string | null;
  has_default: boolean;
  optional?: boolean;
}
interface BlockInfo {
  block_id: string;
  file: string;
  lineno: number;
  label: string;
  category: string;
  params: ParamInfo[];
  outputs: string[];
  output_annotations?: (string | null)[];
  doc: string | null;
}
interface DocNode {
  id: string;
  block: string;
  params: Record<string, any>;
  position: { x: number; y: number };
  notes?: string;
}
interface DocEdge {
  id: string;
  source: string;
  source_output: string;
  target: string;
  target_param: string;
}
interface DocComment {
  id: string;
  title: string;
  description: string;
  position: { x: number; y: number };
  width: number;
  height: number;
  color: string;
}
interface ForgeDoc {
  schema: string;
  name: string;
  nodes: DocNode[];
  edges: DocEdge[];
  comments?: DocComment[];
}
interface NodeState {
  history_hash: string | null;
  stale: boolean;
  problems: string[];
}

type RunPhase = "running" | "done" | "failed" | "blocked" | "cancelled";

interface FigureRef {
  uri: string;
  kind: string; // "image" | "html"
  path?: string; // workspace-relative, for "open in browser"
}
type LightboxState = { items: FigureRef[]; index: number } | null;

interface ProgressInfo {
  current?: number;
  total?: number;
  label?: string;
  percent?: number;
}

/** Desktop comment palette + theme (frontend/src/utils/commentColors.ts). */
const COMMENT_COLORS = [
  "#64748b", "#6366f1", "#3b82f6", "#06b6d4", "#14b8a6", "#22c55e", "#f59e0b", "#f97316", "#f43f5e",
];

function rgba(hex: string, alpha: number): string {
  const match = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!match) return hex;
  const value = parseInt(match[1], 16);
  return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
}

function commentTheme(color: string, selected: boolean) {
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
    resizerLine: rgba(color, 0.82),
  };
}

/** Defensive shape normalization — a malformed doc must degrade to an empty
 * canvas, never crash rendering. */
function normalizeDoc(d: any): ForgeDoc {
  return {
    schema: d?.schema ?? "starforge/1",
    name: typeof d?.name === "string" ? d.name : "",
    nodes: Array.isArray(d?.nodes) ? d.nodes : [],
    edges: Array.isArray(d?.edges) ? d.edges : [],
    comments: Array.isArray(d?.comments) ? d.comments : [],
  };
}

/** Compact display form of an annotation for handle-adjacent labels:
 * "pd.DataFrame | None" → "DataFrame | None". */
function shortAnnotation(annotation: string | null | undefined): string | null {
  if (!annotation) return null;
  const text = annotation
    .replace(/['"]/g, "")
    .split("|")
    .map((part) => part.trim().split("[")[0].split(".").pop() ?? part)
    .join(" | ");
  return text.length > 22 ? text.slice(0, 21) + "…" : text;
}

// ------------------------------------------------------------- canvas node

interface CanvasNodeData {
  label: string;
  category: string;
  params: ParamInfo[];
  outputs: string[];
  connectedParams: string[];
  literals: Record<string, any>;
  outputAnnotations: (string | null)[];
  status: string; // stale | fresh | running | failed | blocked | problem | unknown
  problems: string[];
  errorSummary: string | null;
  figures: FigureRef[];
  onOpenFigures: (figures: FigureRef[], index: number) => void;
  progress: ProgressInfo | null;
  startedAt: number | null;
}

/** Mirrors the desktop BlockNode anatomy: status-tinted header with a status
 * dot and the block name; param/output rows (handles live INSIDE
 * position:relative rows so edge anchors always match the visible row —
 * desktop uses fixed offsets, but our rows carry editable literals); footer
 * with the category icon badge and a status label. */
const STATUS_LABEL: Record<string, string> = {
  unknown: "Not run",
  stale: "Stale",
  running: "Running",
  fresh: "Complete",
  failed: "Error",
  problem: "Error",
  blocked: "Blocked",
  cancelled: "Cancelled",
};

/** On-node figure browser: one thumbnail at a time with ‹ › paging, so
 * multi-figure blocks never overflow the node. Click opens the lightbox. */
function FigureCarousel({
  figures,
  onOpen,
}: {
  figures: FigureRef[];
  onOpen: (figures: FigureRef[], index: number) => void;
}) {
  const [index, setIndex] = useState(0);
  const safe = figures.length > 0 ? ((index % figures.length) + figures.length) % figures.length : 0;
  const figure = figures[safe];
  return (
    <div className="sf-carousel">
      {figure.kind === "image" ? (
        <img
          className="sf-thumb sf-carousel-img"
          src={figure.uri}
          draggable={false}
          title="Click to expand"
          onClick={(e) => {
            e.stopPropagation();
            onOpen(figures, safe);
          }}
        />
      ) : (
        <button
          className="sf-figure-chip"
          onClick={(e) => {
            e.stopPropagation();
            onOpen(figures, safe);
          }}
        >
          ⧉ interactive figure
        </button>
      )}
      {figures.length > 1 && (
        <div className="sf-carousel-bar nodrag" onClick={(e) => e.stopPropagation()}>
          <button onClick={() => setIndex(safe - 1)}>‹</button>
          <span>
            {safe + 1}/{figures.length}
          </span>
          <button onClick={() => setIndex(safe + 1)}>›</button>
        </div>
      )}
    </div>
  );
}

/** Live elapsed readout, ticking locally so the canvas doesn't re-render. */
function Elapsed({ since }: { since: number }) {
  const [, tick] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => tick((t) => t + 1), 500);
    return () => clearInterval(interval);
  }, []);
  return <>{((Date.now() - since) / 1000).toFixed(1)}s</>;
}

function CanvasNode({ data, selected }: NodeProps<CanvasNodeData>) {
  const cat = resolveCategoryStyle(data.category);
  const running = data.status === "running";
  return (
    <div className={`sf-node sf-${data.status}${selected ? " sf-node-selected" : ""}`}>
      <div className={`sf-node-head sf-head-${data.status}`}>
        <span className={`sf-dot sf-${data.status}`} />
        <span className="sf-node-label">{data.label}</span>
      </div>
      <div className="sf-rows">
        {data.params.map((p) => {
          const wired = data.connectedParams.includes(p.name);
          const typeLabel = shortAnnotation(p.annotation);
          return (
            <div key={p.name} className="sf-row sf-row-param">
              <Handle
                type="target"
                position={Position.Left}
                id={p.name}
                className={wired ? "sf-handle sf-wired" : "sf-handle"}
              />
              {typeLabel && (
                <span className="sf-handle-type sf-handle-type-left" title={p.annotation ?? undefined}>
                  {typeLabel}
                </span>
              )}
              <span className="sf-param-name">{p.name}</span>
              {!wired && (
                <span className="sf-param-value">
                  {p.name in data.literals
                    ? JSON.stringify(data.literals[p.name])
                    : p.default_repr ?? (p.optional ? "None" : "required")}
                </span>
              )}
            </div>
          );
        })}
        {data.outputs.map((name, i) => {
          const typeLabel = shortAnnotation(data.outputAnnotations[i]);
          return (
            <div key={name} className="sf-row sf-row-output">
              <span className="sf-output-name">{name}</span>
              <Handle type="source" position={Position.Right} id={name} className="sf-handle" />
              {typeLabel && (
                <span
                  className="sf-handle-type sf-handle-type-right"
                  title={data.outputAnnotations[i] ?? undefined}
                >
                  {typeLabel}
                </span>
              )}
            </div>
          );
        })}
      </div>
      {data.figures.length > 0 && <FigureCarousel figures={data.figures} onOpen={data.onOpenFigures} />}
      {running && (
        <div className="sf-progress">
          <div className="sf-progress-meta">
            <span className="sf-progress-label">{data.progress?.label ?? "Working"}</span>
            <span className="sf-progress-count">
              {data.progress?.total !== undefined
                ? `${data.progress.current ?? 0}/${data.progress.total}`
                : data.progress?.current !== undefined
                  ? `${data.progress.current}`
                  : ""}
            </span>
          </div>
          <div className="sf-progress-track">
            <div
              className={`sf-progress-fill${data.progress?.percent === undefined ? " sf-indeterminate" : ""}`}
              style={{
                width:
                  data.progress?.percent === undefined
                    ? "35%"
                    : `${Math.max(2, Math.round(data.progress.percent * 100))}%`,
              }}
            />
          </div>
        </div>
      )}
      <div className="sf-node-foot">
        <span className="sf-badge" style={{ background: cat.badgeBg, color: cat.badgeText }}>
          <span aria-hidden="true">{cat.icon}</span>
          {data.category}
        </span>
        <span className="sf-status-label">
          {running && data.startedAt ? (
            <>
              Running · <Elapsed since={data.startedAt} />
            </>
          ) : (
            STATUS_LABEL[data.status] ?? data.status
          )}
        </span>
      </div>
      {(data.problems.length > 0 || data.errorSummary) && (
        <div className="sf-problem-strip" title={data.errorSummary ?? data.problems.join("\n")}>
          {data.errorSummary ?? data.problems[0]}
        </div>
      )}
    </div>
  );
}

interface CommentNodeData {
  title: string;
  description: string;
  color: string;
  onChange: (patch: Partial<DocComment>) => void;
}

/** Canvas annotation box, mirroring the desktop CommentNode anatomy: themed
 * translucent wash + colored border, title row with an 18px color-swatch
 * button (opens the swatch panel), separator, description area, themed
 * resizer. (Desktop's custom HSV editor is a later nicety.) */
function CommentNode({ data, selected }: NodeProps<CommentNodeData>) {
  const [menuOpen, setMenuOpen] = useState(false);
  const theme = commentTheme(data.color, selected || menuOpen);
  useEffect(() => {
    if (!selected) setMenuOpen(false);
  }, [selected]);
  return (
    <div
      className="sf-comment"
      style={{ background: theme.background, borderColor: theme.border, boxShadow: theme.shadow }}
    >
      <NodeResizer
        isVisible={selected}
        minWidth={160}
        minHeight={96}
        color={theme.resizerLine}
        onResizeEnd={(_e, params) =>
          data.onChange({
            position: { x: Math.round(params.x), y: Math.round(params.y) },
            width: Math.round(params.width),
            height: Math.round(params.height),
          })
        }
      />
      <div className="sf-comment-head">
        <input
          className="sf-comment-title nodrag nopan"
          defaultValue={data.title}
          placeholder="Comment Title"
          style={{ color: theme.color }}
          onBlur={(e) => e.target.value !== data.title && data.onChange({ title: e.target.value })}
        />
        <button
          className="sf-comment-swatch nodrag nopan"
          title="Comment color"
          style={{ background: theme.buttonFill, borderColor: theme.buttonRing }}
          onClick={(e) => {
            e.stopPropagation();
            setMenuOpen((open) => !open);
          }}
        />
      </div>
      <div className="sf-comment-separator" style={{ background: theme.separator }} />
      <textarea
        className="sf-comment-body nodrag nopan"
        defaultValue={data.description}
        placeholder="Add a description…"
        onBlur={(e) => e.target.value !== data.description && data.onChange({ description: e.target.value })}
      />
      {menuOpen && (
        <div className="sf-comment-menu nodrag nopan" onClick={(e) => e.stopPropagation()}>
          <span className="sf-comment-menu-label">Color</span>
          <div className="sf-comment-swatch-grid">
            {COMMENT_COLORS.map((color) => (
              <button
                key={color}
                className={`sf-comment-swatch-option${color === data.color ? " sf-swatch-active" : ""}`}
                style={{ background: color }}
                onClick={() => {
                  data.onChange({ color });
                  setMenuOpen(false);
                }}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const nodeTypes = { forgeBlock: CanvasNode, forgeComment: CommentNode };

// ---------------------------------------------------------------- previews

function PreviewView({
  preview,
  artifact,
  onOpenFigure,
}: {
  preview: any;
  artifact?: { uri?: string; kind?: string; path?: string };
  onOpenFigure?: (figure: FigureRef) => void;
}) {
  if (!preview) return null;
  if (preview.kind === "figure") {
    if (!artifact?.uri) return <div className="sf-hint">figure (artifact unavailable)</div>;
    const figure: FigureRef = { uri: artifact.uri, kind: artifact.kind ?? "image", path: artifact.path };
    return figure.kind === "image" ? (
      <img
        className="sf-thumb sf-thumb-inspector"
        src={figure.uri}
        draggable={false}
        title="Click to expand"
        onClick={() => onOpenFigure?.(figure)}
      />
    ) : (
      <button className="sf-figure-chip" onClick={() => onOpenFigure?.(figure)}>
        ⧉ open interactive figure
      </button>
    );
  }
  if (preview.kind === "table") {
    return (
      <div className="sf-preview">
        <div className="sf-preview-meta">
          {preview.shape[0]} × {preview.shape[1]}
          {preview.columns_truncated ? " (cropped)" : ""}
        </div>
        <div className="sf-table-scroll">
          <table className="sf-table">
            <thead>
              <tr>
                <th />
                {preview.columns.map((c: string) => (
                  <th key={c}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.rows.map((row: any[], i: number) => (
                <tr key={i}>
                  <td className="sf-table-index">{String(preview.index?.[i] ?? i)}</td>
                  {row.map((cell, j) => (
                    <td key={j}>{String(cell)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }
  if (preview.kind === "array") {
    return (
      <div className="sf-preview">
        <div className="sf-preview-meta">
          ndarray {JSON.stringify(preview.shape)} · {preview.dtype}
        </div>
        <pre className="sf-pre">{JSON.stringify(preview.corner, null, 1)}</pre>
      </div>
    );
  }
  if (preview.kind === "value") {
    return <pre className="sf-pre">{JSON.stringify(preview.value, null, 2)}</pre>;
  }
  return <pre className="sf-pre">{String(preview.text ?? "")}</pre>;
}

// -------------------------------------------------------------------- app

function App() {
  const [doc, setDoc] = useState<ForgeDoc>({ schema: "starforge/1", name: "", nodes: [], edges: [] });
  const [palette, setPalette] = useState<BlockInfo[]>([]);
  const [paletteErrors, setPaletteErrors] = useState<Record<string, string[]>>({});
  const [hashes, setHashes] = useState<Record<string, NodeState>>({});
  const [runPhase, setRunPhase] = useState<Record<string, RunPhase>>({});
  const [failures, setFailures] = useState<Record<string, string>>({});
  const [runId, setRunId] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [manifest, setManifest] = useState<any>(null);
  const [kernelError, setKernelError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [nodeFigures, setNodeFigures] = useState<Record<string, FigureRef[]>>({});
  const [lightbox, setLightbox] = useState<LightboxState>(null);
  const [runProgress, setRunProgress] = useState<Record<string, ProgressInfo>>({});
  const [startedAt, setStartedAt] = useState<Record<string, number>>({});
  const [hoverCard, setHoverCard] = useState<{ block: BlockInfo; top: number } | null>(null);
  const [edgeTip, setEdgeTip] = useState<{ text: string; x: number; y: number } | null>(null);
  const lightboxRef = useRef<LightboxState>(null);
  lightboxRef.current = lightbox;

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const flowRef = useRef<ReactFlowInstance | null>(null);
  const docRef = useRef(doc);
  docRef.current = doc;
  // Last payloads applied, serialized. Self-echoes after commits and
  // identical re-pushes (focus rescans, post-run refreshes) are dropped here
  // before they can touch state — otherwise every interaction triggers a
  // multi-pass full-canvas re-render that reads as a flash.
  const lastSeenRef = useRef<{ doc?: string; palette?: string; hashes?: string; figures?: string }>({});
  // Timestamp of the last drag activity (0 = not dragging). Time-based so a
  // missed dragStop can never permanently freeze doc→canvas syncing.
  const draggingRef = useRef(0);

  const blocksById = useMemo(() => new Map(palette.map((b) => [b.block_id, b])), [palette]);

  // ------------------------------------------------------------ messaging

  useEffect(() => {
    const onMessage = (event: MessageEvent) => {
      const { type, payload } = event.data;
      const seen = lastSeenRef.current;
      if (type === "doc") {
        const normalized = normalizeDoc(payload);
        const serialized = JSON.stringify(normalized);
        if (seen.doc === serialized) return;
        seen.doc = serialized;
        setDoc(normalized);
      } else if (type === "palette") {
        const serialized = JSON.stringify(payload);
        if (seen.palette === serialized) return;
        seen.palette = serialized;
        setPalette(payload.blocks);
        setPaletteErrors(payload.errors ?? {});
        setKernelError(null);
      } else if (type === "hashes") {
        const serialized = JSON.stringify(payload);
        if (seen.hashes === serialized) return;
        seen.hashes = serialized;
        setHashes(payload);
      }
      else if (type === "runStarted") {
        setRunId(payload);
        setRunPhase({});
        setFailures({});
        setRunProgress({});
        setStartedAt({});
      } else if (type === "runEvent") {
        const e = payload;
        if (e.event === "node_started") {
          setRunPhase((s) => ({ ...s, [e.node]: "running" }));
          setStartedAt((s) => ({ ...s, [e.node]: Date.now() }));
        } else if (e.event === "node_progress") {
          setRunProgress((s) => ({
            ...s,
            [e.node]: { current: e.current, total: e.total, label: e.label, percent: e.percent },
          }));
        } else if (e.event === "node_completed") setRunPhase((s) => ({ ...s, [e.node]: "done" }));
        else if (e.event === "node_failed") {
          setRunPhase((s) => ({ ...s, [e.node]: "failed" }));
          setFailures((f) => ({ ...f, [e.node]: e.traceback ?? "failed" }));
        } else if (e.event === "node_blocked") setRunPhase((s) => ({ ...s, [e.node]: "blocked" }));
        else if (e.event === "run_finished") {
          setRunId(null);
          // Nodes still "running" at the end never got a terminal event —
          // the worker was cancelled (or died). Reflect that on the canvas.
          const leftover: RunPhase = e.status === "cancelled" ? "cancelled" : e.status === "failed" ? "failed" : "done";
          setRunPhase((s) => {
            const next: Record<string, RunPhase> = { ...s };
            for (const id of Object.keys(next)) if (next[id] === "running") next[id] = leftover;
            return next;
          });
        }
      } else if (type === "manifest") setManifest(payload);
      else if (type === "nodeFigures") {
        const serialized = JSON.stringify(payload);
        if (seen.figures === serialized) return;
        seen.figures = serialized;
        setNodeFigures(payload);
      } else if (type === "kernelError") setKernelError(payload);
    };
    window.addEventListener("message", onMessage);
    vscode.postMessage({ type: "ready" });
    return () => window.removeEventListener("message", onMessage);
  }, []);

  /** Images expand in the lightbox; interactive HTML opens a dedicated
   * editor-tab webview (iframes can't render webview resources — see the
   * extension's showFigurePanel). */
  const openFigure = useCallback((figures: FigureRef[], index: number) => {
    const figure = figures[index];
    if (figure.kind === "html" && figure.path) {
      vscode.postMessage({
        type: "openFigurePanel",
        payload: { path: figure.path, title: figure.path.split("/").pop() },
      });
    } else {
      setLightbox({ items: figures, index });
    }
  }, []);

  const commit = useCallback((next: ForgeDoc) => {
    // Normalize so our optimistic state serializes identically to its echo
    // (the file round-trip adds defaults like comments: []) — that's what
    // lets lastSeenRef recognize and drop the echo.
    const normalized = normalizeDoc(next);
    // docRef must update SYNCHRONOUSLY: React Flow can invoke several
    // handlers in one event tick (e.g. node-delete + edge-delete), and the
    // second must build on the first's result, not on the pre-render doc.
    docRef.current = normalized;
    lastSeenRef.current.doc = JSON.stringify(normalized);
    setDoc(normalized); // optimistic; the identical echo is dropped on arrival
    vscode.postMessage({ type: "updateDoc", payload: normalized });
  }, []);

  // ------------------------------------------------- doc → React Flow sync

  const statusFor = useCallback(
    (nodeId: string): string => {
      const phase = runPhase[nodeId];
      if (phase === "running") return "running";
      if (phase === "failed") return "failed";
      if (phase === "blocked") return "blocked";
      if (phase === "cancelled") return "cancelled";
      const state = hashes[nodeId];
      if (!state) return "unknown";
      if (state.problems.length > 0) return "problem";
      return state.stale ? "stale" : "fresh";
    },
    [hashes, runPhase],
  );

  useEffect(() => {
    // Never fight an in-flight drag — but a drag with no activity for 1.5s
    // is considered abandoned (missed dragStop) and syncing resumes.
    if (draggingRef.current && Date.now() - draggingRef.current < 1500) return;
    setNodes((current) => {
      const byId = new Map(current.map((n) => [n.id, n]));
      const next = doc.nodes.map((dn) => {
        const info = blocksById.get(dn.block);
        const existing = byId.get(dn.id);
        const failure = failures[dn.id];
        const data: CanvasNodeData = {
          label: info?.label ?? dn.block,
          category: info?.category ?? "?",
          params: info?.params ?? [],
          outputs: info?.outputs ?? ["output"],
          connectedParams: doc.edges.filter((e) => e.target === dn.id).map((e) => e.target_param),
          literals: dn.params,
          outputAnnotations: info?.output_annotations ?? [],
          status: statusFor(dn.id),
          problems: hashes[dn.id]?.problems ?? (info ? [] : [`block '${dn.block}' not found`]),
          errorSummary: failure
            ? failure.trim().split("\n").pop() ?? "failed"
            : runPhase[dn.id] === "cancelled"
              ? "Execution cancelled"
              : null,
          figures: nodeFigures[dn.id] ?? [],
          onOpenFigures: openFigure,
          progress: runProgress[dn.id] ?? null,
          startedAt: startedAt[dn.id] ?? null,
        };
        return {
          id: dn.id,
          type: "forgeBlock",
          position: dn.position ?? { x: 0, y: 0 },
          data,
          // Freshly pasted nodes arrive selected, ready to drag into place.
          selected: existing?.selected ?? pendingSelectRef.current.has(dn.id),
        };
      });
      // Comments render behind blocks; edits round-trip through the doc.
      const commentNodes = (doc.comments ?? []).map((comment) => ({
        id: comment.id,
        type: "forgeComment",
        position: comment.position ?? { x: 0, y: 0 },
        zIndex: -10,
        style: { width: comment.width ?? 280, height: comment.height ?? 140 },
        data: {
          title: comment.title ?? "",
          description: comment.description ?? "",
          color: comment.color ?? COMMENT_COLORS[0],
          onChange: (patch: Partial<DocComment>) => {
            const current = docRef.current;
            commit({
              ...current,
              comments: (current.comments ?? []).map((c) => (c.id === comment.id ? { ...c, ...patch } : c)),
            });
          },
        },
        selected: byId.get(comment.id)?.selected ?? false,
      }));
      const next2 = [...commentNodes, ...next];
      // Consume pending selections HERE, inside the updater — it runs later
      // than the effect body, so clearing outside would race and lose them.
      if (pendingSelectRef.current.size > 0 && next2.some((n) => n.selected)) {
        pendingSelectRef.current = new Set();
      }
      return next2 as Node<any>[];
    });
  }, [doc, blocksById, hashes, runPhase, failures, nodeFigures, runProgress, startedAt, statusFor, setNodes, commit]);

  /** Best-effort static compatibility between an output annotation and a
   * param annotation. Conservative: warn only on a confident mismatch —
   * editors warn, runtimes err (DESIGN.md §2.3). */
  const edgeWarning = useCallback(
    (edge: DocEdge): string | null => {
      const sourceNode = doc.nodes.find((n) => n.id === edge.source);
      const targetNode = doc.nodes.find((n) => n.id === edge.target);
      const sourceInfo = sourceNode ? blocksById.get(sourceNode.block) : undefined;
      const targetInfo = targetNode ? blocksById.get(targetNode.block) : undefined;
      if (!sourceInfo || !targetInfo) return null;
      const outIndex = sourceInfo.outputs.indexOf(edge.source_output);
      const outAnn = sourceInfo.output_annotations?.[outIndex] ?? null;
      const paramAnn = targetInfo.params.find((p) => p.name === edge.target_param)?.annotation ?? null;
      if (!outAnn || !paramAnn) return null;
      const tokens = (annotation: string): string[] =>
        annotation
          .replace(/['"]/g, "")
          .replace(/Optional\[(.*)\]/g, "$1 | None")
          .split("|")
          .map((t) => t.trim())
          .filter(Boolean);
      const tail = (t: string) => t.split("[")[0].split(".").pop()?.trim().toLowerCase() ?? t.toLowerCase();
      const paramTokens = tokens(paramAnn).map(tail);
      const outTokens = tokens(outAnn).map(tail);
      if (paramTokens.includes("any") || paramTokens.includes("object")) return null;
      if (outTokens.includes("any") || outTokens.includes("object")) return null;
      if (outTokens.some((t) => paramTokens.includes(t))) return null;
      return `${edge.target_param} expects ${paramAnn}, receiving ${outAnn}`;
    },
    [doc.nodes, blocksById],
  );

  useEffect(() => {
    setEdges((current) => {
      const selectedIds = new Set(current.filter((e) => e.selected).map((e) => e.id));
      return doc.edges.map((e) => {
        const warning = edgeWarning(e);
        return {
          id: e.id,
          source: e.source,
          sourceHandle: e.source_output,
          target: e.target,
          targetHandle: e.target_param,
          selected: selectedIds.has(e.id),
          className: warning ? "sf-edge-warn" : undefined,
          label: warning ? "⚠" : undefined,
          data: { warning },
        };
      });
    });
  }, [doc.edges, edgeWarning, setEdges]);

  // ------------------------------------------------- interactions → doc

  const onConnect = useCallback(
    (connection: Connection) => {
      if (!connection.source || !connection.target || !connection.targetHandle) return;
      const sourceOutput = connection.sourceHandle ?? "output";
      const current = docRef.current;
      const newEdges = current.edges.filter(
        (e) => !(e.target === connection.target && e.target_param === connection.targetHandle),
      );
      newEdges.push({
        id: `e_${connection.source}_${sourceOutput}_${connection.target}_${connection.targetHandle}`,
        source: connection.source,
        source_output: sourceOutput,
        target: connection.target,
        target_param: connection.targetHandle,
      });
      commit({ ...current, edges: newEdges });
    },
    [commit],
  );

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      if (!flowRef.current) return;
      if (event.dataTransfer.getData("application/starforge-comment")) {
        const at = flowRef.current.screenToFlowPosition({ x: event.clientX, y: event.clientY });
        addCommentAt({ x: at.x - 140, y: at.y - 20 });
        return;
      }
      const blockId = event.dataTransfer.getData("application/starforge-block");
      if (!blockId) return;
      const position = flowRef.current.screenToFlowPosition({ x: event.clientX, y: event.clientY });
      const current = docRef.current;
      const base = (blockId.split(":")[1] ?? "node").replace(/\W/g, "_");
      let i = 1;
      while (current.nodes.some((n) => n.id === `${base}_${i}`)) i++;
      commit({
        ...current,
        nodes: [
          ...current.nodes,
          { id: `${base}_${i}`, block: blockId, params: {}, position: { x: position.x, y: position.y } },
        ],
      });
    },
    [commit],
  );

  const onNodeDragStart = useCallback(() => {
    draggingRef.current = Date.now();
  }, []);

  const onNodeDrag = useCallback(() => {
    draggingRef.current = Date.now(); // keepalive while actively dragging
  }, []);

  /** Multi-select drags move many nodes; React Flow hands us all of them in
   * the third argument — committing only the grabbed node would snap the
   * rest back. */
  const onNodeDragStop = useCallback(
    (_event: unknown, node: Node, draggedNodes?: Node[]) => {
      draggingRef.current = 0;
      const moved = new Map(
        (draggedNodes?.length ? draggedNodes : [node]).map((n) => [
          n.id,
          { x: Math.round(n.position.x), y: Math.round(n.position.y) },
        ]),
      );
      const current = docRef.current;
      commit({
        ...current,
        nodes: current.nodes.map((n) => (moved.has(n.id) ? { ...n, position: moved.get(n.id)! } : n)),
        comments: (current.comments ?? []).map((c) =>
          moved.has(c.id) ? { ...c, position: moved.get(c.id)! } : c,
        ),
      });
    },
    [commit],
  );

  const addCommentAt = useCallback(
    (position: { x: number; y: number }) => {
      const current = docRef.current;
      let i = 1;
      while ((current.comments ?? []).some((c) => c.id === `comment_${i}`)) i++;
      commit({
        ...current,
        comments: [
          ...(current.comments ?? []),
          {
            id: `comment_${i}`,
            title: "Comment",
            description: "",
            position: { x: Math.round(position.x), y: Math.round(position.y) },
            width: 280,
            height: 150,
            color: COMMENT_COLORS[(i - 1) % COMMENT_COLORS.length],
          },
        ],
      });
    },
    [commit],
  );

  /** Delete/Backspace removes the selection (nodes take their edges with
   * them). With Ctrl/Cmd/Shift held, selected nodes are PRUNED instead:
   * their edges are removed but the nodes stay. We own the keybinding
   * (deleteKeyCode={null}) so React Flow never half-applies a deletion. */
  const deleteSelection = useCallback(
    (prune: boolean) => {
      const nodeIds = new Set(
        nodes.filter((n) => n.selected && n.type === "forgeBlock").map((n) => n.id),
      );
      const commentIds = new Set(
        nodes.filter((n) => n.selected && n.type === "forgeComment").map((n) => n.id),
      );
      const edgeIds = new Set(edges.filter((e) => e.selected).map((e) => e.id));
      if (nodeIds.size === 0 && edgeIds.size === 0 && commentIds.size === 0) return;
      const current = docRef.current;
      const touches = (e: DocEdge) =>
        edgeIds.has(e.id) || nodeIds.has(e.source) || nodeIds.has(e.target);
      if (prune) {
        commit({ ...current, edges: current.edges.filter((e) => !touches(e)) });
      } else {
        commit({
          ...current,
          nodes: current.nodes.filter((n) => !nodeIds.has(n.id)),
          edges: current.edges.filter((e) => !touches(e)),
          comments: (current.comments ?? []).filter((c) => !commentIds.has(c.id)),
        });
        setSelected((s) => (s && nodeIds.has(s) ? null : s));
      }
    },
    [nodes, edges, commit],
  );
  const deleteSelectionRef = useRef(deleteSelection);
  deleteSelectionRef.current = deleteSelection;

  // ------------------------------------------- copy / paste (desktop parity)

  const clipboardRef = useRef<{ nodes: DocNode[]; edges: DocEdge[] }>({ nodes: [], edges: [] });
  const pasteDepthRef = useRef(0);
  const pendingSelectRef = useRef<Set<string>>(new Set());

  /** Ctrl/Cmd+C: selected nodes + the edges BETWEEN them. Mirrors desktop
   * App.tsx: native copy wins when text is selected; the system clipboard
   * carries a JSON payload so paste works across canvases. */
  const copySelection = useCallback((): boolean => {
    if (window.getSelection()?.toString()) return false;
    const ids = new Set(nodes.filter((n) => n.selected && n.type === "forgeBlock").map((n) => n.id));
    if (ids.size === 0 && selected) ids.add(selected);
    if (ids.size === 0) return false;
    const current = docRef.current;
    const payload = {
      nodes: current.nodes.filter((n) => ids.has(n.id)).map((n) => JSON.parse(JSON.stringify(n)) as DocNode),
      edges: current.edges
        .filter((e) => ids.has(e.source) && ids.has(e.target))
        .map((e) => ({ ...e })),
    };
    clipboardRef.current = payload;
    pasteDepthRef.current = 0;
    void navigator.clipboard
      ?.writeText(JSON.stringify({ "starforge-clipboard": 1, ...payload }))
      .catch(() => undefined); // in-memory clipboard still works
    return true;
  }, [nodes, selected]);

  /** Ctrl/Cmd+V: prefer the system clipboard (covers cross-canvas paste),
   * fall back to in-memory; stagger repeat pastes by 40px like desktop. */
  const pasteClipboard = useCallback(async () => {
    try {
      const text = await navigator.clipboard?.readText();
      if (text) {
        const parsed = JSON.parse(text);
        if (parsed?.["starforge-clipboard"] === 1 && Array.isArray(parsed.nodes)) {
          clipboardRef.current = { nodes: parsed.nodes, edges: parsed.edges ?? [] };
        }
      }
    } catch {
      // clipboard unreadable (permissions/non-JSON) — use in-memory copy
    }
    const source = clipboardRef.current;
    if (source.nodes.length === 0) return;
    pasteDepthRef.current += 1;
    const offset = 40 * pasteDepthRef.current;
    const current = docRef.current;
    const used = new Set(current.nodes.map((n) => n.id));
    const idMap = new Map<string, string>();
    const pastedNodes = source.nodes.map((n) => {
      const base = (n.block.split(":")[1] ?? "node").replace(/\W/g, "_");
      let i = 1;
      while (used.has(`${base}_${i}`)) i++;
      const id = `${base}_${i}`;
      used.add(id);
      idMap.set(n.id, id);
      return {
        ...(JSON.parse(JSON.stringify(n)) as DocNode),
        id,
        position: { x: (n.position?.x ?? 0) + offset, y: (n.position?.y ?? 0) + offset },
      };
    });
    const pastedEdges = source.edges
      .filter((e) => idMap.has(e.source) && idMap.has(e.target))
      .map((e) => {
        const sourceId = idMap.get(e.source)!;
        const targetId = idMap.get(e.target)!;
        return {
          ...e,
          id: `e_${sourceId}_${e.source_output}_${targetId}_${e.target_param}`,
          source: sourceId,
          target: targetId,
        };
      });
    pendingSelectRef.current = new Set(pastedNodes.map((n) => n.id));
    commit({
      ...current,
      nodes: [...current.nodes, ...pastedNodes],
      edges: [...current.edges, ...pastedEdges],
    });
    setSelected(pastedNodes.length === 1 ? pastedNodes[0].id : null);
  }, [commit]);

  const keyActionsRef = useRef({ copy: copySelection, paste: pasteClipboard });
  keyActionsRef.current = { copy: copySelection, paste: pasteClipboard };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      // Lightbox owns the keyboard while open.
      const box = lightboxRef.current;
      if (box) {
        if (event.key === "Escape") setLightbox(null);
        else if (event.key === "ArrowRight")
          setLightbox({ items: box.items, index: (box.index + 1) % box.items.length });
        else if (event.key === "ArrowLeft")
          setLightbox({ items: box.items, index: (box.index - 1 + box.items.length) % box.items.length });
        event.preventDefault();
        return;
      }
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
        return; // never eat keystrokes from text fields
      }
      const isMod = event.ctrlKey || event.metaKey;
      if (isMod && !event.altKey && !event.shiftKey) {
        const key = event.key.toLowerCase();
        if (key === "c") {
          if (keyActionsRef.current.copy()) event.preventDefault();
          return;
        }
        if (key === "v") {
          event.preventDefault();
          void keyActionsRef.current.paste();
          return;
        }
      }
      if (event.key !== "Delete" && event.key !== "Backspace") return;
      event.preventDefault();
      deleteSelectionRef.current(event.ctrlKey || event.metaKey || event.shiftKey);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  // ------------------------------------------------------------ inspector

  const selectedNode = doc.nodes.find((n) => n.id === selected);
  const selectedInfo = selectedNode ? blocksById.get(selectedNode.block) : undefined;
  const selectedState = selected ? hashes[selected] : undefined;
  const selectedFailure = selected ? failures[selected] : undefined;

  // Auto-load the result preview whenever a fresh node is selected.
  useEffect(() => {
    setManifest(null);
    if (selected && selectedState?.history_hash && !selectedState.stale) {
      vscode.postMessage({ type: "manifest", payload: selectedState.history_hash });
    }
  }, [selected, selectedState?.history_hash, selectedState?.stale]);

  const staleCount = Object.values(hashes).filter((s) => s.stale).length;

  const categories = useMemo(() => {
    const groups = new Map<string, BlockInfo[]>();
    for (const block of palette) {
      if (search && !block.label.toLowerCase().includes(search.toLowerCase())) continue;
      const list = groups.get(block.category) ?? [];
      list.push(block);
      groups.set(block.category, list);
    }
    return [...groups.entries()].sort(([a], [b]) =>
      a === "Built-in" ? -1 : b === "Built-in" ? 1 : a.localeCompare(b),
    );
  }, [palette, search]);

  return (
    <div className="sf-app">
      <aside className="sf-palette">
        <div className="sf-palette-head">
          <span className="sf-title">✱ Blocks</span>
          <input
            className="sf-search"
            placeholder="search…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        {palette.length === 0 && !kernelError && (
          <div className="sf-empty">
            No blocks yet. Decorate a function with <code>@block</code> from <code>starforge</code> and save.
          </div>
        )}
        {kernelError && <div className="sf-error">{kernelError}</div>}
        {categories.map(([category, blocks]) => {
          const cat = resolveCategoryStyle(category);
          return (
            <div key={category} className="sf-category">
              <div className="sf-category-name" style={{ color: cat.accent }}>
                <span aria-hidden="true">{cat.icon}</span> {category}
              </div>
              {blocks.map((block) => (
                <div
                  key={block.block_id}
                  className="sf-palette-block"
                  draggable
                  onDragStart={(e) => {
                    setHoverCard(null);
                    e.dataTransfer.setData("application/starforge-block", block.block_id);
                  }}
                  onMouseEnter={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    setHoverCard({ block, top: Math.min(rect.top, window.innerHeight - 320) });
                  }}
                  onMouseLeave={() => setHoverCard(null)}
                >
                  {block.label}
                </div>
              ))}
            </div>
          );
        })}
        {Object.keys(paletteErrors).length > 0 && (
          <div className="sf-category">
            <div className="sf-category-name">⚠ scan warnings</div>
            {Object.entries(paletteErrors).map(([module, errors]) => (
              <div key={module} className="sf-scan-warning" title={errors.join("\n")}>
                {module}: {errors[0]}
              </div>
            ))}
          </div>
        )}
        <div className="sf-annotations">
          <div className="sf-category-name">Annotations</div>
          <div
            className="sf-comment-palette-item"
            draggable
            title="Drag onto canvas to add a comment annotation"
            onDragStart={(e) => e.dataTransfer.setData("application/starforge-comment", "1")}
          >
            <div className="sf-comment-palette-title">Comment</div>
            <div className="sf-comment-palette-sub">Annotation block</div>
          </div>
        </div>
      </aside>

      {hoverCard && (
        <div className="sf-hover-card" style={{ top: hoverCard.top }}>
          <div className="sf-hover-head">
            <span className="sf-title">{hoverCard.block.label}</span>
            <span
              className="sf-badge"
              style={{
                background: resolveCategoryStyle(hoverCard.block.category).badgeBg,
                color: resolveCategoryStyle(hoverCard.block.category).badgeText,
              }}
            >
              {resolveCategoryStyle(hoverCard.block.category).icon} {hoverCard.block.category}
            </span>
          </div>
          {hoverCard.block.doc && <p className="sf-docstring">{hoverCard.block.doc}</p>}
          {hoverCard.block.params.length > 0 && (
            <>
              <div className="sf-hover-section">inputs</div>
              {hoverCard.block.params.map((p) => (
                <div key={p.name} className="sf-hover-row">
                  <code>{p.name}</code>
                  {p.annotation && <span className="sf-annotation">: {p.annotation}</span>}
                  {p.default_repr !== null && <span className="sf-annotation"> = {p.default_repr}</span>}
                  {p.default_repr === null && p.optional && <span className="sf-annotation"> (optional)</span>}
                </div>
              ))}
            </>
          )}
          <div className="sf-hover-section">outputs</div>
          {hoverCard.block.outputs.map((name, i) => (
            <div key={name} className="sf-hover-row">
              <code>{name}</code>
              {hoverCard.block.output_annotations?.[i] && (
                <span className="sf-annotation">: {hoverCard.block.output_annotations[i]}</span>
              )}
            </div>
          ))}
          {hoverCard.block.file && (
            <div className="sf-hover-source">
              {hoverCard.block.file}:{hoverCard.block.lineno} · drag onto the canvas
            </div>
          )}
        </div>
      )}

      <main
        className="sf-canvas"
        onDrop={onDrop}
        onDragOver={(e) => e.preventDefault()}
        onContextMenu={(e) => e.preventDefault()} // RMB is for panning
      >
        <div className="sf-toolbar">
          <span className="sf-doc-name">{doc.name || "untitled"}</span>
          <span className="sf-summary">
            {doc.nodes.length} nodes · {staleCount} stale
          </span>
          {runId ? (
            <button className="sf-btn sf-cancel" onClick={() => vscode.postMessage({ type: "cancel", payload: runId })}>
              ■ Cancel
            </button>
          ) : (
            <button
              className="sf-btn sf-run"
              disabled={doc.nodes.length === 0}
              onClick={() => vscode.postMessage({ type: "run" })}
            >
              ▶ Run
            </button>
          )}
        </div>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onInit={(instance) => (flowRef.current = instance)}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeDragStart={onNodeDragStart}
          onNodeDrag={onNodeDrag}
          onNodeDragStop={onNodeDragStop}
          onEdgeMouseEnter={(event, edge) => {
            const warning = (edge.data as any)?.warning;
            if (warning) setEdgeTip({ text: warning, x: event.clientX, y: event.clientY });
          }}
          onEdgeMouseLeave={() => setEdgeTip(null)}
          deleteKeyCode={null}
          onNodeClick={(_e, node) => setSelected(node.type === "forgeBlock" ? node.id : null)}
          onPaneClick={() => setSelected(null)}
          // Desktop Forge interaction model (Canvas.tsx): LMB drag draws a
          // selection box; middle/right mouse pans.
          selectionOnDrag
          selectionMode={SelectionMode.Full}
          panOnDrag={[1, 2]}
          fitView
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={18} size={1.2} color="#2a2d3a" />
          <Controls showInteractive={false} />
          <MiniMap
            pannable
            zoomable
            maskColor="rgba(15, 17, 23, 0.7)"
            nodeColor={(n) => {
              const colors: Record<string, string> = {
                stale: "#eab308",
                blocked: "#eab308",
                running: "#3b82f6",
                fresh: "#22c55e",
                failed: "#ef4444",
                problem: "#ef4444",
                cancelled: "#ef4444",
              };
              return colors[(n.data as CanvasNodeData | undefined)?.status ?? ""] ?? "#2a2d3a";
            }}
            // Desktop nav: click anywhere to center there; click a node to jump to it.
            onClick={(_e, position) =>
              flowRef.current?.setCenter(position.x, position.y, {
                zoom: flowRef.current.getZoom(),
                duration: 220,
              })
            }
            onNodeClick={(_e, node) =>
              flowRef.current?.setCenter(
                node.position.x + (node.width ?? 0) / 2,
                node.position.y + (node.height ?? 0) / 2,
                { zoom: flowRef.current.getZoom(), duration: 220 },
              )
            }
          />
        </ReactFlow>
      </main>

      {selectedNode && (
        <aside className="sf-inspector">
          <div className="sf-inspector-head">
            <span className="sf-title">{selectedInfo?.label ?? selectedNode.block}</span>
            {selectedInfo && selectedInfo.file && (
              <button
                className="sf-btn sf-link"
                onClick={() =>
                  vscode.postMessage({
                    type: "openSource",
                    payload: { file: selectedInfo.file, lineno: selectedInfo.lineno },
                  })
                }
              >
                open source ↗
              </button>
            )}
          </div>
          {selectedInfo?.doc && <p className="sf-docstring">{selectedInfo.doc}</p>}
          {(selectedState?.problems ?? []).map((p) => (
            <div key={p} className="sf-error">{p}</div>
          ))}

          <div className="sf-section">Parameters</div>
          {(selectedInfo?.params ?? []).length === 0 && <div className="sf-hint">none</div>}
          {(selectedInfo?.params ?? []).map((p) => {
            const wired = doc.edges.some((e) => e.target === selectedNode.id && e.target_param === p.name);
            return (
              <div key={p.name} className="sf-field">
                <label>
                  {p.name}
                  {p.annotation && <span className="sf-annotation">: {p.annotation}</span>}
                </label>
                {wired ? (
                  <>
                    <span className="sf-wired-tag">via edge</span>
                    {(() => {
                      const edge = doc.edges.find(
                        (e) => e.target === selectedNode.id && e.target_param === p.name,
                      );
                      const warning = edge ? edgeWarning(edge) : null;
                      return warning ? <div className="sf-edge-warning-note">⚠ {warning}</div> : null;
                    })()}
                  </>
                ) : (
                  <input
                    key={`${selectedNode.id}:${p.name}`}
                    defaultValue={
                      p.name in selectedNode.params ? JSON.stringify(selectedNode.params[p.name]) : ""
                    }
                    placeholder={p.default_repr ?? (p.optional ? "None" : "required")}
                    onBlur={(e) => {
                      const text = e.target.value.trim();
                      const params = { ...selectedNode.params };
                      if (!text) delete params[p.name];
                      else {
                        try {
                          params[p.name] = JSON.parse(text);
                        } catch {
                          params[p.name] = text; // bare string convenience
                        }
                      }
                      commit({
                        ...docRef.current,
                        nodes: docRef.current.nodes.map((n) =>
                          n.id === selectedNode.id ? { ...n, params } : n,
                        ),
                      });
                    }}
                  />
                )}
              </div>
            );
          })}

          {selectedFailure && (
            <>
              <div className="sf-section">Error</div>
              <pre className="sf-traceback">{selectedFailure.trim().split("\n").slice(-25).join("\n")}</pre>
            </>
          )}

          <div className="sf-section">Result</div>
          {manifest ? (
            <div className="sf-manifest">
              <div className="sf-preview-meta">⏱ {manifest.duration_seconds}s</div>
              {manifest.outputs?.map((o: any) => (
                <div key={o.name} className="sf-output-block">
                  <div className="sf-output-row">
                    <code>{o.name}</code> · {o.serializer}
                    {o.meta?.shape ? ` · ${JSON.stringify(o.meta.shape)}` : ""}
                    {o.serializer === "ephemeral" && !o.artifact ? " (last run only)" : ""}
                  </div>
                  <PreviewView
                    preview={o.preview}
                    artifact={o.artifact}
                    onOpenFigure={(figure) => openFigure([figure], 0)}
                  />
                </div>
              ))}
              {(manifest.figures ?? []).length > 0 && (
                <>
                  <div className="sf-preview-meta">figures</div>
                  <div className="sf-thumb-grid">
                    {manifest.figures.map((f: any, i: number) =>
                      f.kind === "image" && f.uri ? (
                        <img
                          key={f.file}
                          className="sf-thumb"
                          src={f.uri}
                          draggable={false}
                          title="Click to expand"
                          onClick={() =>
                            openFigure(
                              manifest.figures.map((g: any) => ({ uri: g.uri, kind: g.kind, path: g.path })),
                              i,
                            )
                          }
                        />
                      ) : (
                        <button
                          key={f.file}
                          className="sf-figure-chip"
                          onClick={() => openFigure([{ uri: f.uri, kind: f.kind, path: f.path }], 0)}
                        >
                          ⧉ {f.file}
                        </button>
                      ),
                    )}
                  </div>
                </>
              )}
            </div>
          ) : (
            <div className="sf-hint">
              {selectedState?.stale ? "stale — run to produce a checkpoint" : selectedState ? "loading…" : ""}
            </div>
          )}
        </aside>
      )}

      {edgeTip && (
        <div
          className="sf-edge-tooltip"
          style={{ left: Math.min(edgeTip.x + 12, window.innerWidth - 280), top: edgeTip.y + 14 }}
        >
          ⚠ {edgeTip.text}
        </div>
      )}

      {lightbox && (
        <div className="sf-lightbox" onClick={() => setLightbox(null)}>
          <div className="sf-lightbox-body" onClick={(e) => e.stopPropagation()}>
            {lightbox.items[lightbox.index].kind === "image" ? (
              <img className="sf-lightbox-img" src={lightbox.items[lightbox.index].uri} draggable={false} />
            ) : (
              <div className="sf-lightbox-html-card">
                <span className="sf-lightbox-html-icon">⧉</span>
                <span>Interactive figure</span>
                <button
                  className="sf-btn"
                  onClick={() => {
                    openFigure([lightbox.items[lightbox.index]], 0);
                    setLightbox(null);
                  }}
                >
                  Open interactive panel
                </button>
              </div>
            )}
            <div className="sf-lightbox-bar">
              {lightbox.items.length > 1 && (
                <>
                  <button
                    className="sf-btn sf-lightbox-nav"
                    onClick={() =>
                      setLightbox({
                        items: lightbox.items,
                        index: (lightbox.index - 1 + lightbox.items.length) % lightbox.items.length,
                      })
                    }
                  >
                    ←
                  </button>
                  <span className="sf-lightbox-counter">
                    {lightbox.index + 1} / {lightbox.items.length}
                  </span>
                  <button
                    className="sf-btn sf-lightbox-nav"
                    onClick={() =>
                      setLightbox({ items: lightbox.items, index: (lightbox.index + 1) % lightbox.items.length })
                    }
                  >
                    →
                  </button>
                </>
              )}
              {lightbox.items[lightbox.index].kind === "html" && lightbox.items[lightbox.index].path && (
                <button
                  className="sf-btn sf-lightbox-nav"
                  title="Open in your default browser"
                  onClick={() =>
                    vscode.postMessage({ type: "openExternal", payload: lightbox.items[lightbox.index].path })
                  }
                >
                  ↗ Browser
                </button>
              )}
              <button className="sf-btn sf-lightbox-nav sf-lightbox-close" onClick={() => setLightbox(null)}>
                ✕
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const root = createRoot(document.getElementById("root")!);
root.render(
  <ReactFlowProvider>
    <App />
  </ReactFlowProvider>,
);
