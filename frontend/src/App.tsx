import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import {
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
  type Connection,
  type OnNodesChange,
  type OnEdgesChange,
  type Node,
  type Edge,
  type ReactFlowInstance,
} from "@xyflow/react";
import { exportPipelinePng, exportPipelinePdf } from "./utils/exportCanvas";
import {
  downloadPipelineExport,
  installCustomBlock,
  deleteCustomBlock,
  downloadBlockTemplate,
  exportCustomBlock,
  extractPluginMetadataFromSource,
  getCustomBlockSource,
  listCustomBlocks,
  type InstallBlockResult,
  type CustomBlockEntry,
} from "./api/client";
import { BlockPalette } from "./components/BlockPalette";
import { Canvas } from "./components/Canvas";
import { NodeInspector } from "./components/NodeInspector";
import { OnboardingTour } from "./components/OnboardingTour";
import { OnboardingWelcome } from "./components/OnboardingWelcome";
import { PluginManagerModal } from "./components/PluginManagerModal";
import { Toolbar } from "./components/Toolbar";
import { useAppUpdate } from "./hooks/useAppUpdate";
import { usePipeline, type ForgeNodeData } from "./hooks/usePipeline";
import type { BlockSpec } from "./types/pipeline";
import { isBuiltInCategory } from "./utils/categoryStyles";

const HISTORY_LIMIT = 50;
const GRID_SNAP_SIZE = 50;

interface GraphSnapshot {
  nodes: Node<ForgeNodeData>[];
  edges: Edge[];
}

function graphSignature(nodes: Node<ForgeNodeData>[], edges: Edge[]): string {
  const nodeIds = nodes.map((n) => n.id).sort().join("|");
  const edgeIds = edges.map((e) => e.id).sort().join("|");
  return `${nodeIds}::${edgeIds}`;
}

function cloneValue<T>(value: T): T {
  if (typeof structuredClone === "function") {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value)) as T;
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  if (target.closest("[contenteditable='true']")) return true;
  const tag = target.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select";
}

/** Returns true if the user has a non-empty text selection in the document. */
function hasTextSelection(): boolean {
  const sel = window.getSelection();
  return !!sel && sel.type === "Range" && sel.toString().length > 0;
}

// ── Cross-tab clipboard via localStorage ────────────────────────────────────
const FORGE_CLIPBOARD_KEY = "forge-clipboard-v2";

interface ClipboardEntry {
  id: string;
  nodes: Node<ForgeNodeData>[];
  edges?: Edge[];
}

type PluginBlockSummary = CustomBlockEntry["blocks"][number];

function fallbackPluginTitle(filename: string): string {
  const stem = filename.replace(/\.py$/i, "");
  const parts = stem.split(/[\s_-]+/).filter(Boolean);
  if (parts.length === 0) {
    return "Custom Plugin";
  }
  return parts
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function fallbackPluginDescription(filename: string): string {
  return `Custom block plugin installed from ${filename}.`;
}

function sortPluginBlocks(blocks: PluginBlockSummary[]): PluginBlockSummary[] {
  return [...blocks].sort(
    (a, b) =>
      a.name.localeCompare(b.name, undefined, { sensitivity: "base" }) ||
      a.key.localeCompare(b.key, undefined, { sensitivity: "base" }),
  );
}

function mergePluginBlocks(
  primary: PluginBlockSummary[],
  fallback: PluginBlockSummary[],
): PluginBlockSummary[] {
  const merged = new Map<string, PluginBlockSummary>();
  for (const block of fallback) {
    merged.set(block.key, block);
  }
  for (const block of primary) {
    merged.set(block.key, block);
  }
  return sortPluginBlocks(Array.from(merged.values()));
}

function generateClipboardId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function writeClipboard(nodes: Node<ForgeNodeData>[], edges: Edge[] = []): string {
  const id = generateClipboardId();
  try {
    const entry: ClipboardEntry = { id, nodes, edges };
    localStorage.setItem(FORGE_CLIPBOARD_KEY, JSON.stringify(entry));
  } catch {
    // localStorage full or unavailable — silent fail, in-memory clipboard still works
  }
  return id;
}

function readClipboard(): ClipboardEntry | null {
  try {
    const raw = localStorage.getItem(FORGE_CLIPBOARD_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as ClipboardEntry;
  } catch {
    return null;
  }
}

export default function App() {
  const {
    availableUpdate,
    installing: installingUpdate,
    installUpdate,
  } = useAppUpdate();
  const {
    blocks,
    nodes,
    setNodes,
    edges,
    setEdges,
    pipelineId,
    pipelineFilename,
    pipelineName,
    setPipelineName,
    setPipelineFilename,
    normalizePipelineFilenameDraft,
    selectedNodeId,
    setSelectedNodeId,
    isRunning,
    isStopping,
    isDirty,
    runError,
    addNode,
    addComment,
    deleteNode,
    pasteNodes,
    updateNodeParams,
    newPipelineDraft,
    runPipeline,
    stopPipeline,
    savePipeline,
    prettifyPipeline,
    loadPipeline,
    reloadBlocks,
  } = usePipeline();

  // ── Custom block install flow ────────────────────────────────────────────────

  const [installState, setInstallState] = useState<
    | { phase: "idle" }
    | { phase: "installing"; filename: string }
    | { phase: "conflict"; file: File; existingFilename: string; suggestedFilename: string }
    | { phase: "result"; result: InstallBlockResult; pluginTitle: string }
    | { phase: "error"; message: string }
  >({ phase: "idle" });

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [showPluginManager, setShowPluginManager] = useState(false);
  const [customPlugins, setCustomPlugins] = useState<CustomBlockEntry[]>([]);
  const [pluginsLoading, setPluginsLoading] = useState(false);
  const [pluginsError, setPluginsError] = useState<string | null>(null);

  const hydratePluginsFromSource = useCallback(
    async (plugins: CustomBlockEntry[]) => {
      return Promise.all(
        plugins.map(async (plugin) => {
          const needsMetadataRefresh =
            plugin.title === fallbackPluginTitle(plugin.filename) ||
            plugin.description === fallbackPluginDescription(plugin.filename);
          if (!needsMetadataRefresh) {
            return plugin;
          }

          try {
            const source = await getCustomBlockSource(plugin.filename);
            return {
              ...plugin,
              ...extractPluginMetadataFromSource(source, plugin.filename),
            };
          } catch {
            return plugin;
          }
        }),
      );
    },
    [],
  );

  const pluginBlocksFromRegistry = useMemo(() => {
    const byFilename = new Map<string, PluginBlockSummary[]>();
    for (const spec of blocks) {
      if (!spec.is_custom || !spec.custom_filename) {
        continue;
      }
      const filename = spec.custom_filename;
      const current = byFilename.get(filename) ?? [];
      current.push({
        key: spec.key,
        name: spec.name,
        category: spec.category,
        version: spec.version,
      });
      byFilename.set(filename, current);
    }
    for (const [filename, entries] of byFilename) {
      byFilename.set(filename, sortPluginBlocks(entries));
    }
    return byFilename;
  }, [blocks]);

  const pluginsForManager = useMemo(() => {
    const byFilename = new Map<string, CustomBlockEntry>();

    for (const plugin of customPlugins) {
      byFilename.set(plugin.filename, {
        ...plugin,
        blocks: mergePluginBlocks(
          plugin.blocks,
          pluginBlocksFromRegistry.get(plugin.filename) ?? [],
        ),
      });
    }

    for (const [filename, registryBlocks] of pluginBlocksFromRegistry) {
      if (byFilename.has(filename)) {
        continue;
      }
      byFilename.set(filename, {
        filename,
        stem: filename.replace(/\.py$/i, ""),
        path: filename,
        requirements: [],
        title: fallbackPluginTitle(filename),
        description: fallbackPluginDescription(filename),
        blocks: registryBlocks,
      });
    }

    return Array.from(byFilename.values()).sort(
      (a, b) =>
        a.title.localeCompare(b.title, undefined, { sensitivity: "base" }) ||
        a.filename.localeCompare(b.filename, undefined, { sensitivity: "base" }),
    );
  }, [customPlugins, pluginBlocksFromRegistry]);

  const customCategories = useMemo(() => {
    return [...new Set(
      blocks
        .filter((spec) => spec.is_custom)
        .map((spec) => spec.category)
        .filter((category) => category.trim().length > 0)
        .filter((category) => !isBuiltInCategory(category)),
    )].sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
  }, [blocks]);

  const loadCustomPlugins = useCallback(async () => {
    setPluginsLoading(true);
    setPluginsError(null);
    try {
      const plugins = await listCustomBlocks();
      setCustomPlugins(await hydratePluginsFromSource(plugins));
    } catch (err: unknown) {
      setPluginsError(err instanceof Error ? err.message : String(err));
    } finally {
      setPluginsLoading(false);
    }
  }, [hydratePluginsFromSource]);

  const openPluginManager = useCallback(() => {
    setShowPluginManager(true);
    void loadCustomPlugins();
  }, [loadCustomPlugins]);

  const runInstall = useCallback(
    async (file: File, conflictResolution?: "overwrite" | "rename") => {
      setInstallState({ phase: "installing", filename: file.name });
      try {
        const pluginMetadata = extractPluginMetadataFromSource(
          await file.text(),
          file.name,
        );
        const result = await installCustomBlock(file, conflictResolution);
        if (result.conflict) {
          // Pause and ask the user how to resolve the conflict
          setInstallState({
            phase: "conflict",
            file,
            existingFilename: result.filename,
            suggestedFilename: result.suggested_filename ?? `${result.filename.replace(/\.py$/, "")}_2.py`,
          });
          return;
        }
        setInstallState({
          phase: "result",
          result,
          pluginTitle: pluginMetadata.title,
        });
        if (result.success) {
          reloadBlocks();
          void loadCustomPlugins();
        }
      } catch (err: unknown) {
        setInstallState({
          phase: "error",
          message: err instanceof Error ? err.message : String(err),
        });
      }
    },
    [loadCustomPlugins, reloadBlocks],
  );

  const handleDropBlockFile = useCallback(
    (file: File) => {
      void runInstall(file);
    },
    [runInstall],
  );

  const handleInstallBlockFromFile = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileInputChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) void runInstall(file);
      e.target.value = "";
    },
    [runInstall],
  );

  const [exportToast, setExportToast] = useState<{ title: string; description: string } | null>(null);
  const exportToastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const showExportToast = useCallback((title: string, description: string) => {
    if (exportToastTimerRef.current) clearTimeout(exportToastTimerRef.current);
    setExportToast({ title, description });
    exportToastTimerRef.current = setTimeout(() => {
      setExportToast(null);
      exportToastTimerRef.current = null;
    }, 4000);
  }, []);

  const handleDownloadTemplate = useCallback(() => {
    downloadBlockTemplate()
      .then((filename) => {
        showExportToast("Template downloaded", `${filename} saved to your Downloads folder.`);
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err);
        showExportToast("Template download failed", msg);
      });
  }, [showExportToast]);

  const handleExportBlock = useCallback((spec: BlockSpec) => {
    if (!spec.custom_filename) return;
    const filename = spec.custom_filename;
    exportCustomBlock(filename)
      .then(() => {
        showExportToast("Plugin source exported", `${filename} saved to your Downloads folder.`);
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err);
        showExportToast("Plugin export failed", msg);
      });
  }, [showExportToast]);

  const handleDeleteBlock = useCallback(
    (spec: BlockSpec) => {
      if (!spec.custom_filename) return;
      if (
        !confirm(
          `Uninstall plugin "${spec.custom_filename}"? Any blocks defined in that file will be removed.`,
        )
      ) {
        return;
      }
      void deleteCustomBlock(spec.custom_filename).then(() => {
        reloadBlocks();
        void loadCustomPlugins();
      });
    },
    [loadCustomPlugins, reloadBlocks],
  );

  const handleExportPlugin = useCallback((plugin: CustomBlockEntry) => {
    void exportCustomBlock(plugin.filename);
  }, []);

  const handleDeletePlugin = useCallback(
    (plugin: CustomBlockEntry) => {
      const blockSummary = plugin.blocks.length === 1
        ? `This will remove 1 block from "${plugin.title}".`
        : `This will remove ${plugin.blocks.length} blocks from "${plugin.title}".`;
      if (
        !confirm(
          `Delete plugin "${plugin.title}" (${plugin.filename})?\n\n${blockSummary}`,
        )
      ) {
        return;
      }
      void deleteCustomBlock(plugin.filename).then(() => {
        reloadBlocks();
        void loadCustomPlugins();
      });
    },
    [loadCustomPlugins, reloadBlocks],
  );

  // ── Onboarding ──────────────────────────────────────────────────────────────

  const [showWelcome, setShowWelcome] = useState(() => {
    try {
      return !localStorage.getItem("forge-onboarded");
    } catch {
      return false;
    }
  });
  const [showTour, setShowTour] = useState(false);
  const [runCount, setRunCount] = useState(0);
  const [completedRunCount, setCompletedRunCount] = useState(0);
  const [paramsChangeCount, setParamsChangeCount] = useState(0);
  const [showReplayTourToast, setShowReplayTourToast] = useState(false);
  const [showTutorialHintToast, setShowTutorialHintToast] = useState(false);
  const tutorialHintTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevIsRunningRef = useRef(false);
  const shiftHeldRef = useRef(false);
  const [showGridSnapHint, setShowGridSnapHint] = useState(false);
  const gridSnapHintTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleStartTour = useCallback(() => {
    setShowWelcome(false);
    setShowTour(true);
  }, []);

  const flashTutorialHintToast = useCallback(() => {
    setShowTutorialHintToast(true);
    if (tutorialHintTimerRef.current) {
      window.clearTimeout(tutorialHintTimerRef.current);
    }
    tutorialHintTimerRef.current = window.setTimeout(() => {
      setShowTutorialHintToast(false);
      tutorialHintTimerRef.current = null;
    }, 5000);
  }, []);

  const handleReplayTour = useCallback(() => {
    setShowReplayTourToast(false);
    setShowWelcome(true);
    setShowTour(false);
    setRunCount(0);
    setCompletedRunCount(0);
    setParamsChangeCount(0);
  }, []);

  const handleSkipOnboarding = useCallback(() => {
    setShowWelcome(false);
    flashTutorialHintToast();
    try {
      localStorage.setItem("forge-onboarded", "1");
    } catch { /* localStorage unavailable */ }
  }, [flashTutorialHintToast]);

  const handleSkipTour = useCallback(() => {
    setShowTour(false);
    flashTutorialHintToast();
    try {
      localStorage.setItem("forge-onboarded", "1");
    } catch { /* localStorage unavailable */ }
  }, [flashTutorialHintToast]);

  const handleTourDone = useCallback(() => {
    setShowTour(false);
    setShowReplayTourToast(false);
    try {
      localStorage.setItem("forge-onboarded", "1");
    } catch { /* localStorage unavailable */ }
  }, []);

  useEffect(() => {
    if (showWelcome || showTour) {
      setShowReplayTourToast(false);
    }
  }, [showTour, showWelcome]);

  useEffect(() => {
    return () => {
      if (tutorialHintTimerRef.current) {
        window.clearTimeout(tutorialHintTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    return () => {
      if (gridSnapHintTimerRef.current) {
        window.clearTimeout(gridSnapHintTimerRef.current);
      }
    };
  }, []);

  // Track Shift key held state for grid snapping
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Shift") shiftHeldRef.current = true;
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === "Shift") shiftHeldRef.current = false;
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, []);

  // Wrapped run handler — tracks run count for the onboarding tour
  const handleRunPipeline = useCallback(() => {
    runPipeline();
    setRunCount((c) => c + 1);
  }, [runPipeline]);

  useEffect(() => {
    if (prevIsRunningRef.current && !isRunning && !runError) {
      setCompletedRunCount((count) => count + 1);
    }
    prevIsRunningRef.current = isRunning;
  }, [isRunning, runError]);

  // Wrapped params handler — tracks edits for the onboarding tour
  const handleUpdateNodeParams = useCallback(
    (nodeId: string, params: Record<string, unknown>) => {
      updateNodeParams(nodeId, params);
      setParamsChangeCount((c) => c + 1);
    },
    [updateNodeParams],
  );

  // Canvas refs for export
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const rfInstanceRef = useRef<ReactFlowInstance<any, any> | null>(null);
  const canvasWrapperRef = useRef<HTMLDivElement | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const showExportError = useCallback((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    console.error("Export failed", error);
    setExportError(message);
    setTimeout(() => setExportError(null), 5000);
  }, []);

  const handleExportPng = useCallback(async () => {
    if (!rfInstanceRef.current || !canvasWrapperRef.current || isExporting) return;
    setIsExporting(true);
    try {
      await exportPipelinePng(rfInstanceRef.current, canvasWrapperRef.current, pipelineName);
    } catch (error: unknown) {
      showExportError(error);
    } finally {
      setIsExporting(false);
    }
  }, [pipelineName, isExporting, showExportError]);

  const handleExportPdf = useCallback(async () => {
    if (!rfInstanceRef.current || !canvasWrapperRef.current || isExporting) return;
    setIsExporting(true);
    try {
      await exportPipelinePdf(rfInstanceRef.current, canvasWrapperRef.current, pipelineName);
    } catch (error: unknown) {
      showExportError(error);
    } finally {
      setIsExporting(false);
    }
  }, [pipelineName, isExporting, showExportError]);

  const handleExportBundle = useCallback(
    async (format: "python" | "notebook") => {
      if (isExporting) return;
      setIsExporting(true);
      try {
        const savedPipelineId = await savePipeline();
        if (!savedPipelineId) {
          throw new Error("Pipeline save did not return an id.");
        }
        await downloadPipelineExport(savedPipelineId, format);
      } catch (error: unknown) {
        showExportError(error);
      } finally {
        setIsExporting(false);
      }
    },
    [isExporting, savePipeline, showExportError],
  );

  // Track which block spec is being dragged from the palette
  const [draggingSpec, setDraggingSpec] = useState<BlockSpec | null>(null);
  const [draggingComment, setDraggingComment] = useState(false);
  const historyRef = useRef<GraphSnapshot[]>([]);
  const lastHistorySignatureRef = useRef<string | null>(null);
  const isApplyingUndoRef = useRef(false);
  const clipboardRef = useRef<{ nodes: Node<ForgeNodeData>[]; edges: Edge[] }>({ nodes: [], edges: [] });
  // Tracks the clipboard id that THIS instance last wrote to localStorage.
  // If the stored id differs on paste, another instance wrote it and we must
  // read from localStorage instead of the stale in-memory ref.
  const lastWrittenIdRef = useRef<string | null>(null);
  const pasteDepthRef = useRef(0);

  const clearHistory = useCallback(() => {
    historyRef.current = [];
    lastHistorySignatureRef.current = null;
  }, []);

  const pushHistorySnapshot = useCallback(() => {
    if (isApplyingUndoRef.current) return;
    const signature = graphSignature(nodes, edges);
    if (lastHistorySignatureRef.current === signature) return;
    historyRef.current.push(cloneValue({ nodes, edges }));
    lastHistorySignatureRef.current = signature;
    if (historyRef.current.length > HISTORY_LIMIT) {
      historyRef.current.splice(0, historyRef.current.length - HISTORY_LIMIT);
    }
  }, [nodes, edges]);

  const undoLastStructuralChange = useCallback(() => {
    const previous = historyRef.current.pop();
    if (!previous) return;
    const top =
      historyRef.current.length > 0
        ? historyRef.current[historyRef.current.length - 1]
        : undefined;
    lastHistorySignatureRef.current = top
      ? graphSignature(top.nodes, top.edges)
      : null;
    isApplyingUndoRef.current = true;
    setNodes(previous.nodes);
    setEdges(previous.edges);
    setSelectedNodeId(null);
    window.setTimeout(() => {
      isApplyingUndoRef.current = false;
    }, 0);
  }, [setNodes, setEdges, setSelectedNodeId]);

  // ── React Flow change handlers ──────────────────────────────────────────────

  const onNodesChange: OnNodesChange<Node<ForgeNodeData>> = useCallback(
    (changes) => {
      const removedNodeIds = changes
        .filter((c) => c.type === "remove")
        .map((c) => c.id);
      if (removedNodeIds.length > 0 && !isApplyingUndoRef.current) {
        pushHistorySnapshot();
        const removedIdSet = new Set(removedNodeIds);
        if (selectedNodeId && removedIdSet.has(selectedNodeId)) {
          setSelectedNodeId(null);
        }
      }
      // Snap to grid when Shift is held during drag
      const processedChanges = shiftHeldRef.current
        ? changes.map((change) => {
            if (change.type === "position" && change.position) {
              return {
                ...change,
                position: {
                  x: Math.round(change.position.x / GRID_SNAP_SIZE) * GRID_SNAP_SIZE,
                  y: Math.round(change.position.y / GRID_SNAP_SIZE) * GRID_SNAP_SIZE,
                },
              };
            }
            return change;
          })
        : changes;
      setNodes((ns) => applyNodeChanges(processedChanges, ns));
    },
    [pushHistorySnapshot, selectedNodeId, setNodes, setSelectedNodeId],
  );

  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => {
      const hasEdgeRemoval = changes.some((c) => c.type === "remove");
      if (hasEdgeRemoval && !isApplyingUndoRef.current) {
        pushHistorySnapshot();
      }
      setEdges((es) => applyEdgeChanges(changes, es));
    },
    [pushHistorySnapshot, setEdges],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      pushHistorySnapshot();
      setEdges((es) =>
        addEdge(
          {
            ...connection,
            id: `e_${connection.source}_${connection.target}_${Date.now()}`,
            type: "smoothstep",
            animated: false,
          },
          es,
        ),
      );
    },
    [pushHistorySnapshot, setEdges],
  );

  // ── Drop from palette ───────────────────────────────────────────────────────

  const handleDropBlock = useCallback(
    (spec: BlockSpec, position: { x: number; y: number }) => {
      pushHistorySnapshot();
      const id = addNode(spec, position);
      setSelectedNodeId(id);
      setDraggingSpec(null);
      // Show grid snap hint on first block drop
      try {
        if (!localStorage.getItem("forge-grid-snap-hint-seen")) {
          localStorage.setItem("forge-grid-snap-hint-seen", "1");
          setShowGridSnapHint(true);
          gridSnapHintTimerRef.current = window.setTimeout(() => {
            setShowGridSnapHint(false);
            gridSnapHintTimerRef.current = null;
          }, 6000);
        }
      } catch { /* localStorage unavailable */ }
    },
    [addNode, pushHistorySnapshot, setSelectedNodeId],
  );

  const handleDropComment = useCallback(
    (position: { x: number; y: number }) => {
      pushHistorySnapshot();
      addComment(position);
      setDraggingComment(false);
    },
    [addComment, pushHistorySnapshot],
  );

  const handleDeleteNode = useCallback(
    (nodeId: string) => {
      pushHistorySnapshot();
      deleteNode(nodeId);
    },
    [deleteNode, pushHistorySnapshot],
  );

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (isEditableTarget(event.target)) return;

      if (
        event.key === "?" &&
        event.shiftKey &&
        !event.ctrlKey &&
        !event.metaKey &&
        !event.altKey
      ) {
        const hasSelectedNodes = nodes.some((node) => node.selected);
        if (!showWelcome && !showTour && !selectedNodeId && !hasSelectedNodes) {
          event.preventDefault();
          setShowReplayTourToast(true);
        }
        return;
      }

      const isMod = event.ctrlKey || event.metaKey;
      if (!isMod || event.altKey) return;

      const key = event.key.toLowerCase();

      if (key === "c" && !event.shiftKey) {
        // If the user has text selected, let the browser's native copy work
        if (hasTextSelection()) return;

        const selected = nodes.filter((n) => n.selected);
        const fallback =
          selected.length === 0 && selectedNodeId
            ? nodes.filter((n) => n.id === selectedNodeId)
            : [];
        const toCopy = selected.length > 0 ? selected : fallback;
        if (toCopy.length === 0) return;

        event.preventDefault();
        const clonedNodes = cloneValue(toCopy);
        // Capture internal edges (both endpoints inside the selection)
        const selectedIds = new Set(toCopy.map((n) => n.id));
        const internalEdges = cloneValue(
          edges.filter((e) => selectedIds.has(e.source) && selectedIds.has(e.target)),
        );
        clipboardRef.current = { nodes: clonedNodes, edges: internalEdges };
        pasteDepthRef.current = 0;
        lastWrittenIdRef.current = writeClipboard(clonedNodes, internalEdges);
        return;
      }

      if (key === "v" && !event.shiftKey) {
        // Always check localStorage first. If the stored id differs from the
        // id this instance last wrote, another instance copied more recently
        // and we must use that data instead of the stale in-memory ref.
        const crossTab = readClipboard();
        if (crossTab && crossTab.id !== lastWrittenIdRef.current) {
          // A different instance (or a newer copy from any instance) owns the
          // clipboard — use it and update our in-memory ref.
          clipboardRef.current = { nodes: crossTab.nodes, edges: crossTab.edges ?? [] };
          lastWrittenIdRef.current = crossTab.id;
        } else if (crossTab && clipboardRef.current.nodes.length === 0) {
          // Same instance wrote it but in-memory is empty (e.g. fresh window
          // that hasn't copied yet — load from storage as a fallback).
          clipboardRef.current = { nodes: crossTab.nodes, edges: crossTab.edges ?? [] };
        }
        const source = clipboardRef.current;
        if (source.nodes.length === 0) return;
        event.preventDefault();
        pushHistorySnapshot();
        pasteDepthRef.current += 1;
        const offset = 40 * pasteDepthRef.current;
        const pastedIds = pasteNodes(source.nodes, { x: offset, y: offset });
        // Remap and restore internal edges
        if (source.edges.length > 0 && pastedIds.length === source.nodes.length) {
          const idMap = new Map<string, string>();
          source.nodes.forEach((n, i) => idMap.set(n.id, pastedIds[i]));
          const remappedEdges: Edge[] = source.edges.map((e, i) => ({
            ...e,
            id: `e_paste_${Date.now()}_${i}`,
            source: idMap.get(e.source) ?? e.source,
            target: idMap.get(e.target) ?? e.target,
          }));
          setEdges((es) => [...es, ...remappedEdges]);
        }
        if (
          source.nodes.length === 1 &&
          source.nodes[0].type !== "commentBlock"
        ) {
          setSelectedNodeId(pastedIds[0] ?? null);
        } else {
          setSelectedNodeId(null);
        }
        return;
      }

      if (key === "z" && !event.shiftKey) {
        event.preventDefault();
        undoLastStructuralChange();
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [
    nodes,
    edges,
    selectedNodeId,
    pasteNodes,
    setEdges,
    pushHistorySnapshot,
    setSelectedNodeId,
    showTour,
    showWelcome,
    undoLastStructuralChange,
  ]);

  // ── Derived selected node ───────────────────────────────────────────────────
  // Exclude comment nodes — they use inline editing, no inspector panel needed.

  const selectedNode = selectedNodeId
    ? nodes.find((n) => n.id === selectedNodeId && n.type !== "commentBlock")
    : null;

  // ── Comment z-index: larger area sits further back ──────────────────────────
  // Sort comment nodes by area descending; assign zIndex -(N) … -1 so the
  // largest comment is behind the smallest, all still behind regular blocks.

  const nodesWithCommentZIndex = useMemo(() => {
    const commentNodes = nodes.filter((n) => n.type === "commentBlock");
    // Sort largest → smallest area
    const sorted = [...commentNodes].sort((a, b) => {
      const aArea = (Number(a.style?.width) || 0) * (Number(a.style?.height) || 0);
      const bArea = (Number(b.style?.width) || 0) * (Number(b.style?.height) || 0);
      return bArea - aArea;
    });
    const zMap = new Map<string, number>();
    const n = sorted.length;
    sorted.forEach((node, i) => zMap.set(node.id, -(n - i)));

    return nodes.map((node) => {
      if (node.type !== "commentBlock") return node;
      const zi = zMap.get(node.id) ?? -1;
      return node.zIndex === zi ? node : { ...node, zIndex: zi };
    });
  }, [nodes]);

  // ── New pipeline ────────────────────────────────────────────────────────────

  const handleNewPipeline = useCallback(() => {
    if (isDirty && !confirm("Create a new pipeline? Unsaved changes will be lost.")) {
      return;
    }
    newPipelineDraft();
    clearHistory();
  }, [clearHistory, isDirty, newPipelineDraft]);

  const handleLoadPipeline = useCallback(
    async (id: string) => {
      await loadPipeline(id);
      clearHistory();
      setSelectedNodeId(null);
    },
    [clearHistory, loadPipeline, setSelectedNodeId],
  );

  return (
    <div
      className="h-full w-full min-w-0 flex flex-col bg-forge-bg text-forge-text overflow-hidden"
      style={{ minHeight: "var(--forge-app-height)" }}
    >
      {/* Hidden file input for "Install Block from File…" */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".py"
        className="hidden"
        onChange={handleFileInputChange}
      />

      <Toolbar
        pipelineName={pipelineName}
        pipelineId={pipelineId}
        pipelineFilename={pipelineFilename}
        customCategories={customCategories}
        appUpdate={availableUpdate ? {
          version: availableUpdate.version,
          action: availableUpdate.action,
          isInstalling: installingUpdate,
        } : null}
        isRunning={isRunning}
        isStopping={isStopping}
        isDirty={isDirty}
        runError={exportError ?? runError}
        onNameChange={setPipelineName}
        onFilenameChange={setPipelineFilename}
        onFilenameBlur={normalizePipelineFilenameDraft}
        onSave={savePipeline}
        onPrettify={prettifyPipeline}
        onLoad={handleLoadPipeline}
        onRun={handleRunPipeline}
        onStop={stopPipeline}
        onNewPipeline={handleNewPipeline}
        isExporting={isExporting}
        onExportPng={handleExportPng}
        onExportPdf={handleExportPdf}
        onExportPython={() => {
          void handleExportBundle("python");
        }}
        onExportNotebook={() => {
          void handleExportBundle("notebook");
        }}
        onDownloadTemplate={handleDownloadTemplate}
        onInstallBlock={handleInstallBlockFromFile}
        onManagePlugins={openPluginManager}
        onInstallAppUpdate={() => {
          void installUpdate();
        }}
      />

      <div className="flex flex-1 min-h-0 min-w-0 overflow-hidden">
        <BlockPalette
          blocks={blocks}
          onDragStart={(spec) => {
            setDraggingSpec(spec);
            setDraggingComment(false);
          }}
          onCommentDragStart={() => {
            setDraggingComment(true);
            setDraggingSpec(null);
          }}
          onExportBlock={handleExportBlock}
          onDeleteBlock={handleDeleteBlock}
        />

        <Canvas
          nodes={nodesWithCommentZIndex}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={(id) =>
            setSelectedNodeId((prev) => (prev === id ? null : id))
          }
          onPaneClick={() => setSelectedNodeId(null)}
          onSelectionChange={(selectedNodes) => {
            const regular = selectedNodes.filter((n) => n.type !== "commentBlock");
            setSelectedNodeId(regular.length === 1 ? regular[0].id : null);
          }}
          onDropBlock={handleDropBlock}
          onDropComment={handleDropComment}
          onDropBlockFile={handleDropBlockFile}
          draggingSpec={draggingSpec}
          draggingComment={draggingComment}
          onCanvasReady={(instance, wrapper) => {
            rfInstanceRef.current = instance;
            canvasWrapperRef.current = wrapper;
          }}
        />

        {selectedNode && (
          <NodeInspector
            node={selectedNode as Node<ForgeNodeData>}
            onParamsChange={handleUpdateNodeParams}
            onDelete={handleDeleteNode}
          />
        )}
      </div>

      {showWelcome && (
        <OnboardingWelcome onStartTour={handleStartTour} onSkip={handleSkipOnboarding} />
      )}
      {showTour && (
        <OnboardingTour
          onDone={handleTourDone}
          onSkip={handleSkipTour}
          nodes={nodesWithCommentZIndex}
          edges={edges}
          runCount={runCount}
          completedRunCount={completedRunCount}
          paramsChangeCount={paramsChangeCount}
          isDragging={draggingSpec !== null || draggingComment}
        />
      )}
      {showTutorialHintToast && <TutorialHintToast />}
      {showGridSnapHint && (
        <GridSnapHintToast onDismiss={() => {
          setShowGridSnapHint(false);
          if (gridSnapHintTimerRef.current) {
            window.clearTimeout(gridSnapHintTimerRef.current);
            gridSnapHintTimerRef.current = null;
          }
        }} />
      )}
      {showReplayTourToast && (
        <ReplayTourToast
          onConfirm={handleReplayTour}
          onDismiss={() => setShowReplayTourToast(false)}
        />
      )}

      {/* Block export / template download toast */}
      {exportToast && (
        <ExportSuccessToast
          title={exportToast.title}
          description={exportToast.description}
          onDismiss={() => {
            if (exportToastTimerRef.current) clearTimeout(exportToastTimerRef.current);
            setExportToast(null);
          }}
        />
      )}

      <PluginManagerModal
        open={showPluginManager}
        plugins={pluginsForManager}
        loading={pluginsLoading}
        error={pluginsForManager.length === 0 ? pluginsError : null}
        onInstall={handleInstallBlockFromFile}
        onDownloadTemplate={handleDownloadTemplate}
        onRefresh={() => {
          void loadCustomPlugins();
        }}
        onClose={() => setShowPluginManager(false)}
        onExport={handleExportPlugin}
        onDelete={handleDeletePlugin}
      />

      {/* Block install progress / result modal */}
      {installState.phase !== "idle" && (
        <BlockInstallModal
          state={installState}
          onClose={() => setInstallState({ phase: "idle" })}
          onResolveConflict={(file, resolution) => void runInstall(file, resolution)}
        />
      )}
    </div>
  );
}

function ReplayTourToast({
  onConfirm,
  onDismiss,
}: {
  onConfirm: () => void;
  onDismiss: () => void;
}) {
  return (
    <div className="fixed bottom-5 left-5 z-50 w-[360px] max-w-[calc(100vw-2.5rem)] rounded-xl border border-[#818cf8]/60 bg-forge-surface/95 px-4 py-3 shadow-[0_0_0_1px_rgba(129,140,248,0.22),0_0_24px_rgba(99,102,241,0.22),0_16px_36px_rgba(0,0,0,0.45)] backdrop-blur-sm animate-fade-in-scale">
      <p className="text-sm text-forge-text">
        Do you want to see the tutorial again?
      </p>
      <div className="mt-3 flex gap-2">
        <button
          onClick={onConfirm}
          className="px-3 py-1.5 rounded-lg bg-[#6366f1] hover:bg-[#818cf8] text-white text-sm font-medium transition-[background-color,transform] duration-150 active:scale-[0.98]"
        >
          Yes
        </button>
        <button
          onClick={onDismiss}
          className="px-3 py-1.5 rounded-lg border border-forge-border text-forge-muted hover:text-forge-text hover:border-forge-border-mid text-sm transition-colors duration-150"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
}

// ── Block install modal ────────────────────────────────────────────────────────

function BlockInstallModal({
  state,
  onClose,
  onResolveConflict,
}: {
  state:
    | { phase: "installing"; filename: string }
    | { phase: "conflict"; file: File; existingFilename: string; suggestedFilename: string }
    | { phase: "result"; result: InstallBlockResult; pluginTitle: string }
    | { phase: "error"; message: string };
  onClose: () => void;
  onResolveConflict: (file: File, resolution: "overwrite" | "rename") => void;
}) {
  const isBlocking = state.phase === "installing";

  const title =
    state.phase === "installing" ? "Installing Plugin…" :
    state.phase === "conflict"   ? "File Already Installed" :
    state.phase === "result"     ? (state.result.success ? "Plugin Installed" : "Install Failed") :
    "Install Error";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in"
      onClick={(e) => {
        if (e.target === e.currentTarget && !isBlocking) onClose();
      }}
    >
      <div className="w-full max-w-sm bg-forge-surface border border-forge-border rounded-lg shadow-2xl overflow-hidden animate-fade-in-scale">

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-forge-border">
          <h2 className="text-forge-text font-semibold text-sm">{title}</h2>
          {!isBlocking && (
            <button onClick={onClose} aria-label="Close" className="text-forge-muted hover:text-forge-text transition-colors">
              ✕
            </button>
          )}
        </div>

        {/* Body */}
        <div className="px-5 py-4 space-y-3">
          {state.phase === "installing" && (
            <div className="flex items-center gap-3">
              <span className="inline-block w-3 h-3 rounded-full bg-forge-accent animate-pulse" />
              <p className="text-forge-text text-sm">
                Installing <span className="font-mono text-xs text-forge-muted">{state.filename}</span>…
              </p>
            </div>
          )}

          {state.phase === "conflict" && (
            <>
              <p className="text-sm text-forge-text">
                A file named{" "}
                <span className="font-mono text-xs bg-forge-bg px-1.5 py-0.5 rounded border border-forge-border">
                  {state.existingFilename}
                </span>{" "}
                is already installed. What would you like to do?
              </p>
              <div className="text-xs text-forge-muted bg-forge-bg rounded border border-forge-border p-3 space-y-1">
                <p><span className="text-forge-text font-medium">Overwrite</span> — replace the existing file. Any blocks it defined will be replaced by the new ones.</p>
                <p><span className="text-forge-text font-medium">Keep Both</span> — install the new file as <span className="font-mono">{state.suggestedFilename}</span>.</p>
              </div>
            </>
          )}

          {state.phase === "result" && (
            <>
              <p className={`text-sm font-medium ${state.result.success ? "text-forge-complete" : "text-forge-error"}`}>
                {state.result.success
                  ? `✓ Installed "${state.pluginTitle}".`
                  : `⚠ ${state.result.message}`}
              </p>
              {state.result.installed_packages.length > 0 && (
                <div>
                  <p className="text-[11px] text-forge-muted mb-1">Packages installed:</p>
                  <ul className="space-y-0.5">
                    {state.result.installed_packages.map((pkg) => (
                      <li key={pkg} className="text-xs font-mono text-forge-complete bg-forge-complete/10 px-2 py-0.5 rounded">
                        {pkg}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {state.result.errors.length > 0 && (
                <div>
                  <p className="text-[11px] text-forge-muted mb-1">Errors:</p>
                  <ul className="space-y-0.5">
                    {state.result.errors.map((err, i) => (
                      <li key={i} className="text-xs text-forge-error">{err}</li>
                    ))}
                  </ul>
                </div>
              )}
              {state.result.success && (
                <p className="text-xs text-forge-muted">
                  The plugin is now available in the palette under <span className="text-forge-text font-medium">Plugins</span>.
                </p>
              )}
            </>
          )}

          {state.phase === "error" && (
            <p className="text-sm text-forge-error">{state.message}</p>
          )}
        </div>

        {/* Footer */}
        {state.phase === "conflict" ? (
          <div className="flex items-center justify-end gap-2 px-5 py-3 border-t border-forge-border">
            <button onClick={onClose} className="btn-ghost">
              Cancel
            </button>
            <button
              onClick={() => onResolveConflict(state.file, "rename")}
              className="px-3 py-1.5 rounded bg-forge-accent hover:bg-forge-accent-hover text-white text-sm font-medium transition-colors"
            >
              Keep Both
            </button>
            <button
              onClick={() => onResolveConflict(state.file, "overwrite")}
              className="px-3 py-1.5 rounded bg-forge-error hover:bg-forge-error/90 text-white text-sm font-medium transition-colors"
            >
              Overwrite
            </button>
          </div>
        ) : !isBlocking ? (
          <div className="flex justify-end px-5 py-3 border-t border-forge-border">
            <button onClick={onClose} className="btn-ghost">
              Close
            </button>
          </div>
        ) : null}
      </div>
    </div>
  );
}

// ── Export success toast ───────────────────────────────────────────────────────

function ExportSuccessToast({
  title,
  description,
  onDismiss,
}: {
  title: string;
  description: string;
  onDismiss: () => void;
}) {
  return (
    <div className="fixed bottom-5 right-5 z-50 max-w-[calc(100vw-2.5rem)] rounded-lg border border-forge-complete/40 bg-forge-surface/95 px-4 py-3 shadow-lg shadow-black/35 backdrop-blur-sm animate-fade-in flex items-start gap-3">
      <span className="text-forge-complete text-base leading-none mt-0.5">✓</span>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-forge-text">{title}</p>
        <p className="text-xs text-forge-muted mt-0.5">{description}</p>
      </div>
      <button
        onClick={onDismiss}
        aria-label="Dismiss"
        className="text-forge-muted hover:text-forge-text transition-colors text-[11px] leading-none shrink-0"
      >
        ✕
      </button>
    </div>
  );
}

function TutorialHintToast() {
  return (
    <div className="fixed bottom-5 left-5 z-40 max-w-[calc(100vw-2.5rem)] rounded-lg border border-forge-border bg-forge-surface/95 px-3 py-2 shadow-lg shadow-black/35 backdrop-blur-sm animate-fade-in">
      <p className="text-xs text-forge-muted">
        Didn&apos;t mean to do that?{" "}
        <span className="text-forge-text">Shift + ?</span> to bring the tutorial
        back.
      </p>
    </div>
  );
}

function GridSnapHintToast({ onDismiss }: { onDismiss: () => void }) {
  return (
    <div className="fixed bottom-5 right-5 z-40 flex items-center gap-3 max-w-[calc(100vw-2.5rem)] rounded-lg border border-forge-border bg-forge-surface/95 px-3 py-2 shadow-lg shadow-black/35 backdrop-blur-sm animate-fade-in">
      <p className="text-xs text-forge-muted">
        Tip: Hold <span className="text-forge-text font-medium">Shift</span>{" "}
        while dragging to snap blocks to a {GRID_SNAP_SIZE}px grid.
      </p>
      <button
        onClick={onDismiss}
        aria-label="Dismiss"
        className="text-forge-muted/60 hover:text-forge-muted transition-colors text-xs shrink-0"
      >
        ✕
      </button>
    </div>
  );
}
