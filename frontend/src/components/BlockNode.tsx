import { Suspense, lazy, memo, useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  Handle,
  NodeResizer,
  Position,
  useReactFlow,
  type Node,
  type NodeProps,
  type ResizeParams,
} from "@xyflow/react";
import { checkpointImageUrl, getCheckpointProvenance } from "@/api/client";
import type { ForgeNodeData } from "@/hooks/usePipeline";
import {
  categoryBadgeClass,
  categoryIcon,
  useCategoryStyleVersion,
} from "@/utils/categoryStyles";
import { useNodeOperations } from "./NodeOperationsContext";

// Monaco is only needed when the user actually has a code-typed block on the
// canvas. Lazy-load to keep the initial bundle slim.
const MonacoCodeEditor = lazy(() => import("./MonacoCodeEditor"));

function blockHasCodeParam(data: ForgeNodeData): boolean {
  const schema = data.paramSchema;
  if (!Array.isArray(schema)) return false;
  return schema.some(
    (p) => p.key === "code" || /code/i.test(p.description ?? ""),
  );
}

type ForgeNode = Node<ForgeNodeData, "forgeBlock">;

// Status → border colour
const STATUS_BORDER: Record<string, string> = {
  idle: "border-forge-border",
  stale: "border-forge-stale",
  running: "border-forge-running",
  complete: "border-forge-complete",
  error: "border-forge-error",
};

// Status → header colour
const STATUS_HEADER: Record<string, string> = {
  idle: "bg-forge-border/40",
  stale: "bg-forge-stale/20",
  running: "bg-forge-running/20",
  complete: "bg-forge-complete/20",
  error: "bg-forge-error/20",
};

const STATUS_LABEL: Record<string, string> = {
  idle: "Not run",
  stale: "Stale",
  running: "Running",
  complete: "Complete",
  error: "Error",
};

const STATUS_DOT: Record<string, string> = {
  idle: "bg-forge-muted",
  stale: "bg-forge-stale",
  running: "bg-forge-running animate-pulse",
  complete: "bg-forge-complete",
  error: "bg-forge-error",
};

export const BlockNode = memo(function BlockNode({
  id,
  data,
  selected,
  width,
}: NodeProps<ForgeNode>) {
  useCategoryStyleVersion();
  const { blockName, category, n_inputs, inputLabels, outputLabels, nodeState } = data;
  const { getNode, setNodes } = useReactFlow();
  const nodeOps = useNodeOperations();
  const status = nodeState?.status ?? "idle";
  const border = STATUS_BORDER[status] ?? STATUS_BORDER.idle;
  const header = STATUS_HEADER[status] ?? STATUS_HEADER.idle;
  const badge = categoryBadgeClass(category);
  const icon = categoryIcon(category);

  // Ensure the block is tall enough to contain all input handles.
  // Handles for n_inputs > 1 are positioned at (30 + i*24)px from the top;
  // the last one sits at (30 + (n_inputs-1)*24)px, so add 20px bottom clearance.
  const nOutputs = Math.max(outputLabels?.length ?? 1, 1);
  const maxHandles = Math.max(n_inputs, nOutputs);
  const minHeight =
    maxHandles > 1 ? 30 + (maxHandles - 1) * 24 + 20 : undefined;
  const isViz = category === "Visualization";
  const isCode = blockHasCodeParam(data);
  const minResizableHeight = isCode
    ? Math.max(minHeight ?? 0, 220)
    : isViz
      ? Math.max(minHeight ?? 0, 86)
      : Math.max(minHeight ?? 0, 170);

  const handleCodeChange = useCallback(
    (next: string) => {
      if (!nodeOps) return;
      const currentParams = (data.params ?? {}) as Record<string, unknown>;
      if (currentParams.code === next) return;
      nodeOps.updateNodeParams(id, { ...currentParams, code: next });
    },
    [id, data.params, nodeOps],
  );

  // React Flow listens for arrows/space/etc. via native document listeners,
  // which fire after React's synthetic event system. Stop the native bubble
  // phase at the editor wrapper so keystrokes never reach document.
  const editorWrapperRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!isCode) return;
    const el = editorWrapperRef.current;
    if (!el) return;
    const stop = (e: KeyboardEvent) => {
      e.stopPropagation();
    };
    el.addEventListener("keydown", stop);
    el.addEventListener("keyup", stop);
    el.addEventListener("keypress", stop);
    return () => {
      el.removeEventListener("keydown", stop);
      el.removeEventListener("keyup", stop);
      el.removeEventListener("keypress", stop);
    };
  }, [isCode]);
  const checkpointId = nodeState?.checkpointId;
  const progressCurrent = nodeState?.progressCurrent;
  const progressTotal = nodeState?.progressTotal;
  const progressPercent =
    typeof nodeState?.progressPercent === "number"
      ? Math.max(0, Math.min(1, nodeState.progressPercent))
      : undefined;
  const progressLabel = nodeState?.progressLabel;
  const showProgress =
    status === "running" &&
    (typeof progressCurrent === "number" ||
      typeof progressTotal === "number" ||
      typeof progressPercent === "number" ||
      typeof progressLabel === "string");

  // Fetch image list from provenance when checkpoint arrives for viz blocks
  const [images, setImages] = useState<string[]>([]);
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);
  const [primaryImageAspect, setPrimaryImageAspect] = useState<number | null>(null);
  const prevCheckpointRef = useRef<string | undefined>(undefined);
  const autoSizedCheckpointRef = useRef<string | null>(null);
  const idleNormalizedRef = useRef(false);

  useEffect(() => {
    if (!isViz || !checkpointId || checkpointId === prevCheckpointRef.current) {
      return;
    }
    prevCheckpointRef.current = checkpointId;
    // Clear stale images immediately
    setImages([]);
    setPrimaryImageAspect(null);
    autoSizedCheckpointRef.current = null;
    idleNormalizedRef.current = false;
    let cancelled = false;
    void getCheckpointProvenance(checkpointId)
      .then((prov) => {
        if (!cancelled && prov.images?.length) {
          setImages(prov.images);
        }
      })
      .catch(() => {
        /* provenance unavailable — show nothing */
      });
    return () => {
      cancelled = true;
    };
  }, [isViz, checkpointId]);

  // Clear thumbnails when node goes stale or idle so old images don't linger
  useEffect(() => {
    if (status === "stale" || status === "idle") {
      setImages([]);
      setPrimaryImageAspect(null);
      prevCheckpointRef.current = undefined;
      autoSizedCheckpointRef.current = null;
      idleNormalizedRef.current = false;
    }
  }, [status]);

  // Build input handles for multi-input blocks
  const inputHandles =
    n_inputs === 0
      ? []
      : n_inputs === 1
        ? [{ id: "input_0", label: inputLabels?.[0] ?? "Input 1" }]
        : Array.from({ length: n_inputs }, (_, i) => ({
            id: `input_${i}`,
            label: inputLabels?.[i] ?? `Input ${i + 1}`,
          }));
  const outputHandles =
    nOutputs === 1
      ? [{ id: "output_0", label: outputLabels?.[0] ?? "Output 1" }]
      : Array.from({ length: nOutputs }, (_, i) => ({
          id: `output_${i}`,
          label: outputLabels?.[i] ?? `Output ${i + 1}`,
        }));

  const handleResize = useCallback(
    (_: unknown, params: ResizeParams) => {
      setNodes((ns) =>
        ns.map((n) =>
          n.id === id
            ? {
                ...n,
                position: { x: params.x, y: params.y },
                style: {
                  ...n.style,
                  width: params.width,
                  height: params.height,
                },
              }
            : n,
        ),
      );
    },
    [id, setNodes],
  );

  const hasError = status === "error" && Boolean(nodeState?.errorMessage);

  useEffect(() => {
    if (!isViz) return;
    if (checkpointId || images.length > 0 || status === "running") return;
    if (idleNormalizedRef.current) return;
    const existing = getNode(id);
    const hasSizedStyle =
      typeof existing?.style?.width === "number" ||
      typeof existing?.style?.height === "number";
    if (hasSizedStyle) {
      idleNormalizedRef.current = true;
      return;
    }

    const compactWidth = 170;
    const compactHeight = Math.max(minHeight ?? 0, 86);
    idleNormalizedRef.current = true;
    setNodes((ns) =>
      ns.map((n) =>
        n.id === id
          ? {
              ...n,
              style: {
                ...n.style,
                width: compactWidth,
                height: compactHeight,
              },
            }
          : n,
      ),
    );
  }, [checkpointId, getNode, id, images.length, isViz, minHeight, setNodes, status]);

  useEffect(() => {
    if (!isViz || !checkpointId || !primaryImageAspect) return;
    if (autoSizedCheckpointRef.current === checkpointId) return;

    const baseNodeWidth =
      typeof getNode(id)?.style?.width === "number"
        ? Number(getNode(id)?.style?.width)
        : typeof width === "number"
          ? width
          : 220;

    const safeNodeWidth = Math.max(140, Math.round(baseNodeWidth));
    const innerImageWidth = Math.max(180, safeNodeWidth - 20);
    const imageHeight = innerImageWidth / primaryImageAspect;
    const chromeHeight = 76 + (showProgress ? 34 : 0) + (hasError ? 24 : 0);
    const nextHeight = Math.max(minResizableHeight, Math.round(chromeHeight + imageHeight));

    autoSizedCheckpointRef.current = checkpointId;
    setNodes((ns) =>
      ns.map((n) =>
        n.id === id
          ? {
              ...n,
              style: {
                ...n.style,
                width: safeNodeWidth,
                height: nextHeight,
              },
            }
          : n,
      ),
    );
  }, [
    checkpointId,
    getNode,
    hasError,
    id,
    isViz,
    minResizableHeight,
    primaryImageAspect,
    setNodes,
    showProgress,
    width,
  ]);

  const singleImageMode = images.length === 1;

  return (
    <>
      <div
        className={`
          relative rounded-lg border-2 ${border}
          ${selected ? "ring-2 ring-forge-accent ring-offset-1 ring-offset-forge-bg shadow-xl shadow-black/50" : "shadow-lg shadow-black/40 hover:shadow-xl hover:shadow-black/50"}
          bg-forge-surface cursor-default
          flex flex-col
          transition-[border-color,box-shadow] duration-200
          ${isViz || isCode ? "min-w-[140px] w-full h-full" : "min-w-[170px] max-w-[220px]"}
        `}
        style={minHeight !== undefined ? { minHeight } : undefined}
      >
        {(isViz || isCode) && (
          <NodeResizer
            isVisible={selected}
            minWidth={isCode ? 320 : 140}
            minHeight={minResizableHeight}
            keepAspectRatio={isViz && primaryImageAspect !== null}
            onResize={handleResize}
            color="#6366f1"
            handleStyle={{
              width: 9,
              height: 9,
              borderRadius: 2,
              background: "#4f46e5",
              border: "1.5px solid #818cf8",
            }}
            lineStyle={{
              borderColor: "#6366f1",
              borderWidth: 1,
              borderStyle: "dashed",
            }}
          />
        )}

        {/* Input handles — position the Handle directly so its DOM center
            (where React Flow attaches edges) matches the visual dot. */}
        {inputHandles.map((h, i) => {
          const top = n_inputs > 1 ? `${30 + i * 24}px` : "50%";
          return (
            <div key={h.id}>
              <Handle
                id={h.id}
                type="target"
                position={Position.Left}
                style={{
                  top,
                  background: "#6366f1",
                  border: "2px solid #818cf8",
                  width: 10,
                  height: 10,
                }}
                title={h.label}
              />
              <span
                className="absolute w-[82px] text-[9px] text-right text-forge-muted leading-tight pointer-events-none"
                style={{ top, left: -90, transform: "translateY(-50%)" }}
              >
                {h.label}
              </span>
            </div>
          );
        })}

        {/* Header */}
        <div className={`${header} rounded-t-md px-3 py-2`}>
          <div className="flex items-center gap-2">
            <span
              className={`inline-block w-2 h-2 rounded-full flex-shrink-0 ${STATUS_DOT[status]}`}
            />
            <span className="text-forge-text font-semibold text-sm leading-tight break-words min-w-0">
              {blockName}
            </span>
          </div>
        </div>

        {/* Inline code editor for FreeformCode-style blocks */}
        {isCode && (
          <div
            ref={editorWrapperRef}
            className="nodrag nopan nowheel px-2 pt-2 pb-1 flex-1 min-h-0"
          >
            <Suspense
              fallback={
                <div className="h-full w-full rounded border border-forge-border bg-forge-bg flex items-center justify-center text-[11px] text-forge-muted">
                  loading editor…
                </div>
              }
            >
              <MonacoCodeEditor
                value={typeof data.params?.code === "string" ? (data.params.code as string) : ""}
                onChange={handleCodeChange}
                height="100%"
                fontSize={8}
                ariaLabel="cell code"
              />
            </Suspense>
          </div>
        )}

        {/* Footer */}
        <div className="px-3 py-1.5 flex items-center justify-between gap-2">
          <span
            className={`inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 rounded ${badge}`}
          >
            <span aria-hidden="true">{icon}</span>
            {category}
          </span>
          <span className="text-[10px] text-forge-muted">
            {STATUS_LABEL[status]}
          </span>
        </div>

        {showProgress && (
          <div className="px-3 pb-2">
            <div className="mb-1 flex items-center justify-between gap-2 text-[9px] text-forge-muted">
              <span className="truncate">
                {progressLabel && progressLabel.trim().length > 0
                  ? progressLabel
                  : "Working"}
              </span>
              <span className="font-mono">
                {typeof progressTotal === "number"
                  ? `${progressCurrent ?? 0}/${progressTotal}`
                  : `${progressCurrent ?? 0}`}
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded bg-forge-border/70">
              <div
                className={`h-full bg-forge-running ${progressPercent === undefined ? "animate-pulse" : "transition-[width] duration-300 ease-out"}`}
                style={{
                  width:
                    progressPercent === undefined
                      ? "35%"
                      : `${Math.max(2, Math.round(progressPercent * 100))}%`,
                }}
              />
            </div>
          </div>
        )}

        {/* Error tooltip */}
        {status === "error" && nodeState?.errorMessage && (
          <div className="px-3 pb-2">
            <p
              className="text-[10px] text-forge-error truncate"
              title={nodeState.errorMessage}
            >
              {nodeState.errorMessage}
            </p>
          </div>
        )}

        {/* Inline image thumbnails for visualization blocks */}
        {isViz && images.length > 0 && singleImageMode && (
          <div className="px-2 pb-2 flex-1 min-h-0">
            {images.map((filename) => {
              const url = checkpointId
                ? checkpointImageUrl(checkpointId, filename)
                : null;
              if (!url) return null;
              return (
                <button
                  key={filename}
                  className="
                    nodrag nopan
                    block w-full h-full rounded overflow-hidden
                    border border-forge-border hover:border-forge-accent
                    transition-[border-color,box-shadow] duration-150 cursor-zoom-in
                    hover:shadow-md hover:shadow-forge-accent/10
                  "
                  onClick={(e) => {
                    e.stopPropagation();
                    setLightboxSrc(url);
                  }}
                  title="Click to expand"
                >
                  <img
                    src={url}
                    alt={filename}
                    loading="lazy"
                    className="block w-full h-full object-contain"
                    onLoad={(e) => {
                      const img = e.currentTarget;
                      if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                        setPrimaryImageAspect(img.naturalWidth / img.naturalHeight);
                      }
                    }}
                    draggable={false}
                  />
                </button>
              );
            })}
          </div>
        )}

        {isViz && images.length > 1 && (
          <div className="px-2 pb-2 space-y-1.5">
            {images.map((filename) => {
              const url = checkpointId
                ? checkpointImageUrl(checkpointId, filename)
                : null;
              if (!url) return null;
              return (
                <button
                  key={filename}
                  className="
                    nodrag nopan
                    block w-full rounded overflow-hidden
                    border border-forge-border hover:border-forge-accent
                    transition-[border-color,box-shadow] duration-150 cursor-zoom-in
                    hover:shadow-md hover:shadow-forge-accent/10
                  "
                  onClick={(e) => {
                    e.stopPropagation();
                    setLightboxSrc(url);
                  }}
                  title="Click to expand"
                >
                  <img
                    src={url}
                    alt={filename}
                    loading="lazy"
                    className="block w-full object-contain"
                    draggable={false}
                  />
                </button>
              );
            })}
          </div>
        )}

        {/* Output handles — position the Handle directly so its DOM center
            (where React Flow attaches edges) matches the visual dot. */}
        {outputHandles.map((h, i) => {
          const top = nOutputs > 1 ? `${30 + i * 24}px` : "50%";
          return (
            <div key={h.id}>
              <Handle
                id={h.id}
                type="source"
                position={Position.Right}
                style={{
                  top,
                  background: "#6366f1",
                  border: "2px solid #818cf8",
                  width: 10,
                  height: 10,
                }}
                title={h.label}
              />
              <span
                className="absolute w-[82px] text-[9px] text-left text-forge-muted leading-tight pointer-events-none"
                style={{ top, right: -90, transform: "translateY(-50%)" }}
              >
                {h.label}
              </span>
            </div>
          );
        })}
      </div>

      {/* Lightbox — portalled to document.body so it escapes the RF transform */}
      {lightboxSrc &&
        createPortal(
          <NodeLightbox
            src={lightboxSrc}
            onClose={() => setLightboxSrc(null)}
          />,
          document.body,
        )}
    </>
  );
});

// ── Lightbox ──────────────────────────────────────────────────────────────────

function NodeLightbox({
  src,
  onClose,
}: {
  src: string;
  onClose: () => void;
}) {
  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm animate-fade-in"
      onClick={onClose}
    >
      <div
        className="relative bg-forge-surface border border-forge-border rounded-lg shadow-2xl overflow-hidden max-w-[92vw] max-h-[92vh] animate-fade-in-scale"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          className="absolute top-2 right-2 z-10 w-7 h-7 flex items-center justify-center rounded-full bg-forge-bg/80 text-forge-muted hover:text-forge-text transition-colors text-sm"
          onClick={onClose}
          aria-label="Close"
        >
          ✕
        </button>
        <img
          src={src}
          alt="Visualization output"
          className="max-w-[92vw] max-h-[92vh] object-contain"
          draggable={false}
        />
      </div>
    </div>
  );
}
