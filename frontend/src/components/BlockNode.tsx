import { memo, useCallback, useEffect, useRef, useState } from "react";
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
  const minResizableHeight = isViz
    ? Math.max(minHeight ?? 0, 86)
    : Math.max(minHeight ?? 0, 170);
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
  const runningHeightExpandedRef = useRef(false);

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

  // Interactive HTML artifacts (plotly figures) don't have natural dimensions
  // the way an <img> does, so seed a sensible default aspect to trigger the
  // auto-size effect below. The value is roughly matched to matplotlib's
  // default figsize=[10, 6] so an interactive plot opens at the same size as
  // its static counterpart.
  useEffect(() => {
    if (!isViz || images.length === 0 || primaryImageAspect !== null) return;
    if (images[0].toLowerCase().endsWith(".html")) {
      setPrimaryImageAspect(1.6);
    }
  }, [images, primaryImageAspect, isViz]);

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

  // When the block enters the running state, make sure its explicit height
  // can house the progress bar — otherwise the progress strip overflows past
  // the bottom border. Only applies to blocks that already have a fixed
  // style.height (viz blocks, or any block the user has resized); blocks
  // without an explicit height grow naturally with their content.
  useEffect(() => {
    if (status !== "running") {
      runningHeightExpandedRef.current = false;
      return;
    }
    if (runningHeightExpandedRef.current) return;
    const node = getNode(id);
    const currentHeight =
      typeof node?.style?.height === "number"
        ? Number(node.style.height)
        : null;
    if (currentHeight === null) return;
    // Header (~36) + footer (~28) + progress (~34) + error (~24 when set) + buffer.
    const runningMinHeight = Math.max(
      minResizableHeight,
      130 + (hasError ? 24 : 0),
    );
    if (currentHeight >= runningMinHeight) return;
    runningHeightExpandedRef.current = true;
    setNodes((ns) =>
      ns.map((n) =>
        n.id === id
          ? { ...n, style: { ...n.style, height: runningMinHeight } }
          : n,
      ),
    );
  }, [status, hasError, id, getNode, setNodes, minResizableHeight]);

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
          ${isViz ? "min-w-[140px] w-full h-full" : "min-w-[170px] max-w-[220px]"}
        `}
        style={minHeight !== undefined ? { minHeight } : undefined}
      >
        {isViz && (
          <NodeResizer
            isVisible={selected}
            minWidth={140}
            minHeight={minResizableHeight}
            keepAspectRatio={primaryImageAspect !== null}
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

        {/* Input handles */}
        {inputHandles.map((h, i) => (
          <div
            key={h.id}
            className="absolute left-0"
            style={{ top: n_inputs > 1 ? `${30 + i * 24}px` : "50%", transform: "translateY(-50%)" }}
          >
            <span className="absolute -left-[90px] w-[82px] text-[9px] text-right text-forge-muted leading-tight pointer-events-none">
              {h.label}
            </span>
            <Handle
              id={h.id}
              type="target"
              position={Position.Left}
              style={{
                background: "#6366f1",
                border: "2px solid #818cf8",
                width: 10,
                height: 10,
              }}
              title={h.label}
            />
          </div>
        ))}

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
              const interactive = filename.toLowerCase().endsWith(".html");
              if (interactive) {
                // Live iframe — plotly handles hover/pan/zoom directly.
                // The expand-to-lightbox affordance lives on a small corner
                // button so it doesn't compete with the chart for clicks.
                return (
                  <div
                    key={filename}
                    className="
                      nodrag nopan nowheel
                      relative block w-full h-full rounded overflow-hidden
                      border border-forge-border hover:border-forge-accent
                      transition-[border-color,box-shadow] duration-150
                      hover:shadow-md hover:shadow-forge-accent/10
                    "
                  >
                    <iframe
                      src={url}
                      title={filename}
                      sandbox="allow-scripts"
                      className="block w-full h-full bg-white"
                      style={{ border: 0 }}
                    />
                    <ExpandOverlayButton
                      onClick={() => setLightboxSrc(url)}
                      title="Expand"
                    />
                  </div>
                );
              }
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
                        setPrimaryImageAspect(
                          img.naturalWidth / img.naturalHeight,
                        );
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
              const interactive = filename.toLowerCase().endsWith(".html");
              if (interactive) {
                return (
                  <div
                    key={filename}
                    className="
                      nodrag nopan nowheel
                      relative block w-full rounded overflow-hidden
                      border border-forge-border hover:border-forge-accent
                      transition-[border-color,box-shadow] duration-150
                      hover:shadow-md hover:shadow-forge-accent/10
                    "
                  >
                    <iframe
                      src={url}
                      title={filename}
                      sandbox="allow-scripts"
                      className="block w-full bg-white"
                      style={{ height: 160, border: 0 }}
                    />
                    <ExpandOverlayButton
                      onClick={() => setLightboxSrc(url)}
                      title="Expand"
                    />
                  </div>
                );
              }
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

        {/* Output handles */}
        {outputHandles.map((h, i) => (
          <div
            key={h.id}
            className="absolute right-0"
            style={{ top: nOutputs > 1 ? `${30 + i * 24}px` : "50%", transform: "translateY(-50%)" }}
          >
            <span className="absolute -right-[90px] w-[82px] text-[9px] text-left text-forge-muted leading-tight pointer-events-none">
              {h.label}
            </span>
            <Handle
              id={h.id}
              type="source"
              position={Position.Right}
              style={{
                background: "#6366f1",
                border: "2px solid #818cf8",
                width: 10,
                height: 10,
              }}
              title={h.label}
            />
          </div>
        ))}
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

// ── Inline expand affordance ──────────────────────────────────────────────────

function ExpandOverlayButton({
  onClick,
  title,
}: {
  onClick: () => void;
  title: string;
}) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      className="
        nodrag nopan nowheel
        absolute top-1 right-1 z-10
        w-6 h-6 flex items-center justify-center
        rounded bg-forge-bg/80 hover:bg-forge-bg text-forge-muted hover:text-forge-text
        border border-forge-border hover:border-forge-accent
        text-xs leading-none
        transition-[background-color,color,border-color] duration-150
      "
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
    >
      ⛶
    </button>
  );
}

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

  const interactive = src.toLowerCase().endsWith(".html");

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
        {interactive ? (
          <iframe
            src={src}
            title="Interactive visualization"
            sandbox="allow-scripts"
            className="bg-white"
            style={{ width: "85vw", height: "85vh", border: 0 }}
          />
        ) : (
          <img
            src={src}
            alt="Visualization output"
            className="max-w-[92vw] max-h-[92vh] object-contain"
            draggable={false}
          />
        )}
      </div>
    </div>
  );
}
