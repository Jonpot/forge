import { Suspense, lazy, useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { getCheckpointPreview, getCheckpointProvenance } from "@/api/client";
import { FileBrowser } from "./FileBrowser";
import { ImageLightbox, ImageThumbnail } from "./ImagePreview";

// Lazy-loaded so Monaco's ~3MB chunk is only fetched when the user opens an
// inspector with a code-typed param. Keeps the main bundle slim.
const MonacoCodeEditor = lazy(() => import("./MonacoCodeEditor"));
import type { ForgeNodeData } from "@/hooks/usePipeline";
import type { BlockParamSpec, BrowseMode, CheckpointPreview } from "@/types/pipeline";
import type { Node } from "@xyflow/react";
import {
  categoryBadgeClass,
  categoryIcon,
  useCategoryStyleVersion,
} from "@/utils/categoryStyles";

interface NodeInspectorProps {
  node: Node<ForgeNodeData>;
  onParamsChange: (nodeId: string, params: Record<string, unknown>) => void;
  onDelete: (nodeId: string) => void;
}

export function NodeInspector({
  node,
  onParamsChange,
  onDelete,
}: NodeInspectorProps) {
  useCategoryStyleVersion();
  const { data } = node;
  const {
    blockName,
    category,
    description,
    inputLabels = [],
    outputLabels = [],
    params,
    paramSchema = [],
    paramDescriptions = {},
    nodeState,
  } = data;
  const checkpointId = nodeState?.checkpointId;

  const [localParams, setLocalParams] = useState<Record<string, unknown>>(
    () => ({ ...params }),
  );
  const [preview, setPreview] = useState<CheckpointPreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [lightboxImage, setLightboxImage] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"params" | "data" | "images">(
    "params",
  );

  // Sync local params when node changes
  useEffect(() => {
    setLocalParams(materializeParams(params, paramSchema));
  }, [node.id, params, paramSchema]);

  // Load preview when checkpoint becomes available
  useEffect(() => {
    if (!checkpointId) {
      setPreview(null);
      return;
    }
    setPreviewLoading(true);
    setPreviewError(null);
    getCheckpointPreview(checkpointId)
      .then((p) => {
        setPreview(p);
        setPreviewLoading(false);
      })
      .catch((err: unknown) => {
        setPreviewError(
          err instanceof Error ? err.message : "Failed to load preview",
        );
        setPreviewLoading(false);
      });
  }, [checkpointId]);

  const handleParamChange = (key: string, value: unknown) => {
    const next = { ...localParams, [key]: value };
    setLocalParams(next);
    onParamsChange(node.id, next);
  };

  const status = nodeState?.status ?? "idle";
  const categoryBadge = categoryBadgeClass(category);
  const categorySymbol = categoryIcon(category);

  // Determine available images from provenance (stored in checkpoint)
  // We rely on checkpointId being present; images listed in preview.dtypes won't include images,
  // but we can speculatively try to detect image presence from the checkpoint's provenance.
  // For now we show a tab if status is complete (images may or may not exist).
  const hasImages = status === "complete" || status === "stale";

  return (
    <aside className="w-72 flex-shrink-0 bg-forge-surface border-l border-forge-border flex flex-col overflow-hidden animate-slide-in-left">
      {/* Header */}
      <div className="px-4 py-3 border-b border-forge-border">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 className="text-forge-text font-semibold text-sm break-words">
              {blockName}
            </h2>
            <p className="mt-1">
              <span
                className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${categoryBadge}`}
              >
                <span aria-hidden="true">{categorySymbol}</span>
                {category}
              </span>
            </p>
            {description && (
              <p className="text-forge-muted text-[11px] mt-1.5 leading-snug">
                {description}
              </p>
            )}
          </div>
          <button
            onClick={() => onDelete(node.id)}
            className="flex-shrink-0 text-forge-muted hover:text-forge-error text-xs px-2 py-1 rounded border border-forge-border hover:border-forge-error transition-[color,border-color] duration-150 active:scale-95"
            title="Delete node"
          >
            Delete
          </button>
        </div>

        {/* Status badge */}
        <div className="mt-3 flex items-center gap-2">
          <StatusBadge status={status} />
          {nodeState?.mode === "reused" && (
            <span className="text-[10px] text-forge-muted">(cached)</span>
          )}
        </div>

        {checkpointId && (
          <p className="mt-1.5 text-[10px] font-mono text-forge-muted truncate">
            {checkpointId}
          </p>
        )}
        <div className="mt-2.5 text-[10px] text-forge-muted space-y-0.5">
          <div className="break-words">
            Inputs: {inputLabels.length ? inputLabels.join(", ") : "None"}
          </div>
          <div className="break-words">
            Outputs: {outputLabels.length ? outputLabels.join(", ") : "output"}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-forge-border">
        {(["params", "data", "images"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 py-2.5 text-xs font-medium transition-[color,border-color] duration-200 ${
              activeTab === tab
                ? "text-forge-accent border-b-2 border-forge-accent"
                : "text-forge-muted hover:text-forge-text border-b-2 border-transparent"
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === "params" && (
        <ParamForm
          params={localParams}
          paramSchema={paramSchema}
          paramDescriptions={paramDescriptions}
          onChange={handleParamChange}
          hideCodeField={paramSchema.some(
            (p) => p.key === "code" || /code/i.test(p.description ?? ""),
          )}
        />
        )}
        {activeTab === "data" && (
          <DataPreview
            preview={preview}
            loading={previewLoading}
            error={previewError}
            hasCheckpoint={!!checkpointId}
          />
        )}
        {activeTab === "images" && (
          <ImagesTab
            checkpointId={checkpointId}
            hasImages={hasImages}
            onOpenLightbox={setLightboxImage}
          />
        )}
      </div>

      {lightboxImage && checkpointId && (
        <ImageLightbox
          checkpointId={checkpointId}
          filename={lightboxImage}
          onClose={() => setLightboxImage(null)}
        />
      )}
    </aside>
  );
}

// ── Param form ────────────────────────────────────────────────────────────────

interface ParamFieldMeta {
  key: string;
  type: string;
  value: unknown;
  required: boolean;
  description: string;
  example: unknown;
  browseMode?: BrowseMode | null;
}

function materializeParams(
  params: Record<string, unknown>,
  paramSchema: BlockParamSpec[],
): Record<string, unknown> {
  const next = { ...params };
  for (const param of paramSchema) {
    if (!(param.key in next)) {
      next[param.key] = param.default ?? null;
    }
  }
  return next;
}

function buildParamFields(
  params: Record<string, unknown>,
  paramSchema: BlockParamSpec[],
  paramDescriptions: Record<string, string>,
): ParamFieldMeta[] {
  const schemaFields = paramSchema.map((param) => ({
    key: param.key,
    type: param.type,
    value: param.key in params ? params[param.key] : param.default,
    required: param.required,
    description: param.description || paramDescriptions[param.key] || "",
    example: param.example,
    browseMode: param.browse_mode,
  }));
  const knownKeys = new Set(schemaFields.map((field) => field.key));
  const extraFields = Object.entries(params)
    .filter(([key]) => !knownKeys.has(key))
    .map(([key, value]) => ({
      key,
      type: typeof value === "undefined" ? "" : typeof value,
      value,
      required: false,
      description: paramDescriptions[key] ?? "",
      example: undefined,
      browseMode: undefined,
    }));
  return [...schemaFields, ...extraFields];
}

function ParamForm({
  params,
  paramSchema,
  paramDescriptions,
  onChange,
  hideCodeField = false,
}: {
  params: Record<string, unknown>;
  paramSchema: BlockParamSpec[];
  paramDescriptions: Record<string, string>;
  onChange: (key: string, value: unknown) => void;
  hideCodeField?: boolean;
}) {
  const allFields = buildParamFields(params, paramSchema, paramDescriptions);
  const fields = hideCodeField
    ? allFields.filter(
        (f) => !(f.key === "code" || /code/i.test(f.description ?? "")),
      )
    : allFields;

  if (fields.length === 0) {
    return (
      <p className="text-forge-muted text-xs px-4 py-6 text-center">
        {hideCodeField ? "Code is edited inline on the block." : "No parameters"}
      </p>
    );
  }

  return (
    <div className="px-4 py-3 space-y-3">
      {fields.map((field) => (
        <ParamField
          key={field.key}
          paramKey={field.key}
          paramType={field.type}
          value={field.value}
          required={field.required}
          description={field.description}
          example={field.example}
          browseMode={field.browseMode}
          onChange={onChange}
        />
      ))}
    </div>
  );
}

function ParamField({
  paramKey,
  paramType,
  value,
  required,
  description,
  example,
  browseMode,
  onChange,
}: {
  paramKey: string;
  paramType: string;
  value: unknown;
  required: boolean;
  description: string;
  example: unknown;
  browseMode?: BrowseMode | null;
  onChange: (key: string, value: unknown) => void;
}) {
  const [showFileBrowser, setShowFileBrowser] = useState(false);
  const label = paramKey
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
  const typeText = normalizeParamType(paramType);
  const placeholder = examplePlaceholder(example);
  const inputId = `param-${paramKey}`;
  const fileBrowserMode = browseMode ?? null;

  const inputClass =
    "w-full bg-forge-bg border border-forge-border rounded px-2 py-1.5 text-forge-text text-xs focus:outline-none focus:border-forge-accent transition-colors";
  const header = (
    <div className="flex items-center gap-2 mb-1">
      <label
        htmlFor={inputId}
        className="block text-forge-muted text-[11px] cursor-pointer"
        title={description || undefined}
      >
        {label}
      </label>
      {required && (
        <span className="rounded bg-forge-stale/20 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-forge-stale">
          Required
        </span>
      )}
      {typeText && (
        <span className="text-[10px] font-mono text-forge-muted">
          {paramType}
        </span>
      )}
    </div>
  );
  const descriptionBlock = (
    <>
      {description && (
        <p className="text-[10px] text-forge-muted mb-1.5 leading-snug">
          {description}
        </p>
      )}
      {!description && placeholder && (
        <p className="text-[10px] text-forge-muted mb-1.5 leading-snug">
          Example: <span className="font-mono">{placeholder}</span>
        </p>
      )}
      {description && placeholder && (
        <p className="text-[10px] text-forge-muted -mt-1 mb-1.5 leading-snug">
          Example: <span className="font-mono">{placeholder}</span>
        </p>
      )}
    </>
  );

  if (isBooleanParam(typeText, value)) {
    const checked = Boolean(value);
    return (
      <div>
        <div className="mb-1">{header}</div>
        <div className="flex items-center justify-between gap-2">
          <span className="text-forge-text text-xs">
            {required ? "Enabled" : "Toggle"}
          </span>
          <button
            role="switch"
            aria-checked={checked}
            onClick={() => onChange(paramKey, !checked)}
            className={`w-8 h-4 rounded-full transition-colors flex items-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forge-accent focus-visible:ring-offset-1 focus-visible:ring-offset-forge-surface ${
              checked ? "bg-forge-accent" : "bg-forge-border"
            }`}
          >
            <div
              className={`w-3 h-3 rounded-full bg-white mx-0.5 transition-transform ${
                checked ? "translate-x-4" : "translate-x-0"
              }`}
            />
          </button>
        </div>
        <div className="mt-1">{descriptionBlock}</div>
      </div>
    );
  }

  if (isNumericParam(typeText, value)) {
    const step = isIntegerParam(typeText) ? 1 : "any";
    return (
      <div>
        {header}
        {descriptionBlock}
        <input
          id={inputId}
          type="number"
          className={inputClass}
          step={step}
          value={value == null ? "" : String(value)}
          placeholder={placeholder || undefined}
          onChange={(e) => {
            const raw = e.target.value.trim();
            if (raw === "") {
              onChange(paramKey, null);
              return;
            }
            const nextValue = isIntegerParam(typeText)
              ? Number.parseInt(raw, 10)
              : Number.parseFloat(raw);
            onChange(paramKey, Number.isNaN(nextValue) ? value : nextValue);
          }}
        />
      </div>
    );
  }

  if (isListParam(typeText, value)) {
    return (
      <div>
        {header}
        {descriptionBlock}
        <ListParamInput
          inputId={inputId}
          inputClass={inputClass}
          value={value}
          placeholder={placeholder || "comma-separated values"}
          typeText={typeText}
          onChange={(parsed) => onChange(paramKey, parsed)}
        />
      </div>
    );
  }

  const isCodeField = paramKey === "code" || /code/i.test(description);
  if (isCodeField) {
    const codeText = value == null ? "" : String(value);
    return (
      <div>
        {header}
        {descriptionBlock}
        <Suspense
          fallback={
            <textarea
              id={inputId}
              className={`${inputClass} min-h-[160px] font-mono resize-y`}
              value={codeText}
              placeholder="loading editor…"
              spellCheck={false}
              readOnly
            />
          }
        >
          <MonacoCodeEditor
            value={codeText}
            onChange={(next) => onChange(paramKey, next === "" ? null : next)}
            ariaLabel={`${paramKey} code editor`}
          />
        </Suspense>
      </div>
    );
  }

  // Default: string / null
  return (
    <div>
      {header}
      {descriptionBlock}
      <div className={fileBrowserMode ? "flex gap-1" : ""}>
        <input
          id={inputId}
          type="text"
          className={`${inputClass} ${fileBrowserMode ? "flex-1 min-w-0" : ""}`}
          value={value == null ? "" : String(value)}
          placeholder={placeholder || (value == null ? "(null)" : undefined)}
          autoCapitalize="off"
          autoCorrect="off"
          spellCheck={false}
          onChange={(e) => {
            const v = e.target.value === "" ? null : e.target.value;
            onChange(paramKey, v);
          }}
        />
        {fileBrowserMode && (
          <button
            type="button"
            onClick={() => setShowFileBrowser(true)}
            className="flex-shrink-0 px-2 py-1.5 rounded bg-forge-bg border border-forge-border text-forge-muted hover:text-forge-text hover:border-forge-accent text-xs transition-colors"
            title={
              fileBrowserMode === "directory"
                ? "Browse folders"
                : fileBrowserMode === "save_file"
                  ? "Browse file output"
                  : "Browse files"
            }
            aria-label={
              fileBrowserMode === "directory"
                ? "Browse folders"
                : fileBrowserMode === "save_file"
                  ? "Browse file output"
                  : "Browse files"
            }
          >
            📂
          </button>
        )}
      </div>
      {showFileBrowser &&
        createPortal(
          <FileBrowser
            initialPath={value == null ? "" : String(value)}
            mode={fileBrowserMode}
            onSelect={(path) => {
              onChange(paramKey, path);
              setShowFileBrowser(false);
            }}
            onClose={() => setShowFileBrowser(false)}
          />,
          document.body,
        )}
    </div>
  );
}

/**
 * A text input for list/tuple params that keeps a local draft string so users
 * can type intermediate values like "10, " without the field rejecting them.
 * Commits the parsed value on blur or Enter.
 */
function ListParamInput({
  inputId,
  inputClass,
  value,
  placeholder,
  typeText,
  onChange,
}: {
  inputId: string;
  inputClass: string;
  value: unknown;
  placeholder: string;
  typeText: string;
  onChange: (parsed: unknown[]) => void;
}) {
  const renderedValue = Array.isArray(value) ? value : [];
  const canonical = renderedValue.join(", ");
  const [draft, setDraft] = useState(canonical);
  const prevCanonical = useRef(canonical);

  // Sync draft when the value changes externally (e.g. node switch)
  useEffect(() => {
    if (canonical !== prevCanonical.current) {
      setDraft(canonical);
      prevCanonical.current = canonical;
    }
  }, [canonical]);

  const commit = useCallback(() => {
    const parts = draft
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    const parsed = coerceListValues(parts, typeText);
    onChange(parsed);
    // Normalise the draft to the parsed representation
    const newCanonical = parsed.join(", ");
    setDraft(newCanonical);
    prevCanonical.current = newCanonical;
  }, [draft, typeText, onChange]);

  return (
    <input
      id={inputId}
      type="text"
      className={inputClass}
      value={draft}
      placeholder={placeholder}
      autoCapitalize="off"
      autoCorrect="off"
      spellCheck={false}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          commit();
        }
      }}
    />
  );
}

function normalizeParamType(paramType: string): string {
  return paramType.toLowerCase().replace(/\s+/g, "");
}

function isBooleanParam(typeText: string, value: unknown): boolean {
  return typeof value === "boolean" || (!typeText.includes("str") && typeText.includes("bool"));
}

function isNumericParam(typeText: string, value: unknown): boolean {
  return (
    typeof value === "number" ||
    (!typeText.includes("str") &&
      (typeText.includes("int") || typeText.includes("float")))
  );
}

function isIntegerParam(typeText: string): boolean {
  return (
    !typeText.includes("str") &&
    typeText.includes("int") &&
    !typeText.includes("float")
  );
}

function isListParam(typeText: string, value: unknown): boolean {
  return Array.isArray(value) || typeText.startsWith("list[") || typeText === "list";
}

function coerceListValues(parts: string[], typeText: string): unknown[] {
  if (typeText.includes("float")) {
    return parts.map((part) => {
      const parsed = Number.parseFloat(part);
      return Number.isNaN(parsed) ? part : parsed;
    });
  }
  if (typeText.includes("int")) {
    return parts.map((part) => {
      const parsed = Number.parseInt(part, 10);
      return Number.isNaN(parsed) ? part : parsed;
    });
  }
  return parts;
}

function examplePlaceholder(example: unknown): string | null {
  if (typeof example === "undefined" || example === null) {
    return null;
  }
  if (Array.isArray(example)) {
    return example.join(", ");
  }
  if (typeof example === "object") {
    try {
      return JSON.stringify(example);
    } catch {
      return null;
    }
  }
  return String(example);
}

// ── Data preview ──────────────────────────────────────────────────────────────

function DataPreview({
  preview,
  loading,
  error,
  hasCheckpoint,
}: {
  preview: CheckpointPreview | null;
  loading: boolean;
  error: string | null;
  hasCheckpoint: boolean;
}) {
  if (!hasCheckpoint) {
    return (
      <p className="text-forge-muted text-xs px-4 py-6 text-center">
        Run the pipeline to see data
      </p>
    );
  }
  if (loading) {
    return (
      <p className="text-forge-muted text-xs px-4 py-6 text-center animate-pulse">
        Loading…
      </p>
    );
  }
  if (error) {
    return (
      <p className="text-forge-error text-xs px-4 py-4 text-center">{error}</p>
    );
  }
  if (!preview) return null;

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-2 border-b border-forge-border flex items-center justify-between">
        <span className="text-forge-muted text-[11px]">
          {preview.total_rows.toLocaleString()} rows × {preview.columns.length} cols
        </span>
        <span className="text-forge-muted text-[11px]">
          Showing {Math.min(preview.rows.length, 50)}
        </span>
      </div>
      <div className="overflow-auto flex-1">
        <table className="w-full text-[10px] border-collapse">
          <thead className="sticky top-0 bg-forge-surface">
            <tr>
              {preview.columns.map((col) => (
                <th
                  key={col}
                  className="px-2 py-1.5 text-left text-forge-muted border-b border-forge-border font-medium whitespace-nowrap"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.rows.map((row, i) => (
              <tr
                key={i}
                className={`hover:bg-forge-accent/5 transition-colors ${i % 2 === 0 ? "" : "bg-forge-bg/30"}`}
              >
                {preview.columns.map((col) => {
                  const v = row[col];
                  return (
                    <td
                      key={col}
                      className="px-2 py-1 border-b border-forge-border/50 text-forge-text whitespace-nowrap font-mono"
                    >
                      {v == null
                        ? <span className="text-forge-muted">null</span>
                        : typeof v === "number"
                          ? Number.isInteger(v)
                            ? v
                            : (v as number).toFixed(4)
                          : String(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Images tab ────────────────────────────────────────────────────────────────

function ImagesTab({
  checkpointId,
  hasImages,
  onOpenLightbox,
}: {
  checkpointId?: string;
  hasImages: boolean;
  onOpenLightbox: (filename: string) => void;
}) {
  const [images, setImages] = useState<string[]>([]);
  const [tried, setTried] = useState(false);

  useEffect(() => {
    if (!checkpointId) {
      setImages([]);
      setTried(false);
      return;
    }
    let cancelled = false;
    setImages([]);
    setTried(false);
    void getCheckpointProvenance(checkpointId)
      .then((prov) => {
        if (cancelled) return;
        if (Array.isArray(prov.images)) {
          setImages(prov.images);
        }
        setTried(true);
      })
      .catch(() => {
        if (!cancelled) {
          setTried(true);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [checkpointId]);

  if (!checkpointId) {
    return (
      <p className="text-forge-muted text-xs px-4 py-6 text-center">
        Run the pipeline to see images
      </p>
    );
  }

  if (!tried) {
    return (
      <p className="text-forge-muted text-xs px-4 py-6 text-center animate-pulse">
        Loading…
      </p>
    );
  }

  if (images.length === 0) {
    return (
      <p className="text-forge-muted text-xs px-4 py-6 text-center">
        {hasImages ? "No images produced by this block" : "Run the pipeline to see images"}
      </p>
    );
  }

  return (
    <div className="p-3 space-y-2">
      {images.map((filename) => (
        <ImageThumbnail
          key={filename}
          checkpointId={checkpointId}
          filename={filename}
          onClick={onOpenLightbox}
        />
      ))}
    </div>
  );
}

// ── Status badge ──────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: string }) {
  const cfg: Record<string, { bg: string; text: string; label: string }> = {
    idle: { bg: "bg-forge-border/60", text: "text-forge-muted", label: "Not run" },
    stale: { bg: "bg-forge-stale/20", text: "text-forge-stale", label: "Stale" },
    running: { bg: "bg-forge-running/20", text: "text-forge-running", label: "Running" },
    complete: { bg: "bg-forge-complete/20", text: "text-forge-complete", label: "Complete" },
    error: { bg: "bg-forge-error/20", text: "text-forge-error", label: "Error" },
  };
  const c = cfg[status] ?? cfg.idle;
  return (
    <span className={`text-[11px] px-2 py-0.5 rounded font-medium ${c.bg} ${c.text}`}>
      {c.label}
    </span>
  );
}
