import { useCallback, useEffect, useRef, useState } from "react";
import type { Node, Edge } from "@xyflow/react";
import {
  fetchBlocks,
  getStaleness,
  cancelPipelineExecution,
  createPipeline,
  deletePipeline as apiDeletePipeline,
  prettifyPipeline as apiPrettifyPipeline,
  updatePipeline,
  getPipeline,
} from "@/api/client";
import { useWebSocket } from "./useWebSocket";
import {
  DEFAULT_COMMENT_COLOR,
  normalizeCommentColor,
} from "@/utils/commentColors";
import type {
  BlockSpec,
  BlockParamSpec,
  CommentItem,
  PipelineGroup,
  NodeState,
  NodeStatus,
  Pipeline,
  PipelineNode,
  WsMessage,
} from "@/types/pipeline";

// Data stored on comment nodes
export interface CommentNodeData extends Record<string, unknown> {
  title: string;
  description: string;
  color?: string | null;
  managed?: boolean;
  groupId?: string | null;
}

// Extra data we hang on each React Flow node
export interface ForgeNodeData extends Record<string, unknown> {
  blockKey: string;
  blockName: string;
  category: string;
  description: string;
  n_inputs: number;
  inputLabels: string[];
  outputLabels: string[];
  params: Record<string, unknown>;
  paramSchema: BlockParamSpec[];
  paramDescriptions: Record<string, string>;
  notes?: string | null;
  groupIds: string[];
  nodeState: NodeState;
}

function makeRfId(nodeId: string) {
  return nodeId;
}

function parseTargetInputFromHandle(targetHandle: string | null | undefined): number | null {
  if (!targetHandle) return null;
  const match = targetHandle.match(/(\d+)$/);
  if (!match) return null;
  const parsed = Number.parseInt(match[1], 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseSourceOutputFromHandle(sourceHandle: string | null | undefined): number | null {
  if (!sourceHandle) return null;
  const match = sourceHandle.match(/(\d+)$/);
  if (!match) return null;
  const parsed = Number.parseInt(match[1], 10);
  return Number.isFinite(parsed) ? parsed : null;
}

const DEFAULT_COMMENT_WIDTH = 300;
const DEFAULT_COMMENT_HEIGHT = 150;
const DEFAULT_BLOCK_WIDTH = 200;
const DEFAULT_BLOCK_HEIGHT = 170;
const DEFAULT_PIPELINE_FILENAME = "untitled_pipeline";

function slugifyPipelineFilename(value: string): string {
  const withoutExtension = value.trim().replace(/\.json$/i, "");
  const cleaned = withoutExtension
    .replace(/[^a-zA-Z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .toLowerCase();
  return cleaned || "pipeline";
}

function readNodeWidth(node: Node<ForgeNodeData>): number {
  if (typeof node.width === "number") return node.width;
  if (typeof node.style?.width === "number") return node.style.width;
  return node.type === "commentBlock" ? DEFAULT_COMMENT_WIDTH : DEFAULT_BLOCK_WIDTH;
}

function readNodeHeight(node: Node<ForgeNodeData>): number {
  if (typeof node.height === "number") return node.height;
  if (typeof node.style?.height === "number") return node.style.height;
  return node.type === "commentBlock" ? DEFAULT_COMMENT_HEIGHT : DEFAULT_BLOCK_HEIGHT;
}

function resolveCommentGroupId(commentNode: Node<ForgeNodeData>): string {
  const data = commentNode.data as unknown as CommentNodeData;
  if (typeof data.groupId === "string" && data.groupId.trim()) {
    return data.groupId;
  }
  return commentNode.id;
}

function deriveGroupsFromComments(
  commentNodes: Node<ForgeNodeData>[],
  existingGroups: PipelineGroup[],
): PipelineGroup[] {
  const existingById = new Map(existingGroups.map((group) => [group.id, group]));
  return commentNodes.map((commentNode) => {
    const data = commentNode.data as unknown as CommentNodeData;
    const groupId = resolveCommentGroupId(commentNode);
    const existing = existingById.get(groupId);
    return {
      id: groupId,
      name: data.title?.trim() || existing?.name || groupId,
      description: data.description?.trim() || existing?.description || "",
      comment_id: commentNode.id,
    };
  });
}

function computeGroupMemberships(
  regularNodes: Node<ForgeNodeData>[],
  commentNodes: Node<ForgeNodeData>[],
): Map<string, string[]> {
  const memberships = new Map<string, string[]>();
  const orderedComments = [...commentNodes].sort((left, right) => {
    const leftArea = readNodeWidth(left) * readNodeHeight(left);
    const rightArea = readNodeWidth(right) * readNodeHeight(right);
    if (leftArea !== rightArea) return leftArea - rightArea;
    return left.id.localeCompare(right.id);
  });

  for (const node of regularNodes) {
    const centerX = node.position.x + readNodeWidth(node) / 2;
    const centerY = node.position.y + readNodeHeight(node) / 2;
    const groupIds: string[] = [];

    for (const comment of orderedComments) {
      const left = comment.position.x;
      const top = comment.position.y;
      const right = left + readNodeWidth(comment);
      const bottom = top + readNodeHeight(comment);
      if (centerX >= left && centerX <= right && centerY >= top && centerY <= bottom) {
        groupIds.push(resolveCommentGroupId(comment));
      }
    }

    memberships.set(node.id, groupIds);
  }
  return memberships;
}

function buildPipelinePayload(
  nodes: Node<ForgeNodeData>[],
  edges: Edge[],
  name: string,
  groups: PipelineGroup[],
): Pipeline {
  const regularNodes = nodes.filter((n) => n.type !== "commentBlock");
  const commentNodes = nodes.filter((n) => n.type === "commentBlock");
  const derivedGroups = deriveGroupsFromComments(commentNodes, groups);
  const memberships = computeGroupMemberships(regularNodes, commentNodes);

  const comments: CommentItem[] = commentNodes.map((n) => {
    const cd = n.data as unknown as CommentNodeData;
    return {
      id: n.id,
      title: cd.title ?? "",
      description: cd.description ?? "",
      color: normalizeCommentColor(typeof cd.color === "string" ? cd.color : null) ?? DEFAULT_COMMENT_COLOR,
      position: n.position,
      width: readNodeWidth(n),
      height: readNodeHeight(n),
      managed: Boolean(cd.managed),
      group_id: resolveCommentGroupId(n),
    };
  });

  return {
    name,
    nodes: regularNodes.map((n) => ({
      id: n.id,
      block: n.data.blockKey,
      params: n.data.params,
      notes: n.data.notes ?? null,
      group_ids: memberships.get(n.id) ?? [],
      position: n.position,
      ...(typeof n.style?.width === "number" ? { width: n.style.width } : {}),
      ...(typeof n.style?.height === "number" ? { height: n.style.height } : {}),
    })),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      source_output: parseSourceOutputFromHandle(e.sourceHandle),
      sourceHandle: e.sourceHandle ?? null,
      target_input: parseTargetInputFromHandle(e.targetHandle),
      targetHandle: e.targetHandle ?? null,
    })),
    comments,
    groups: derivedGroups,
  };
}

function payloadSignature(pipeline: Pipeline): string {
  return JSON.stringify(pipeline);
}

const INITIAL_PIPELINE_SIGNATURE = payloadSignature({
  name: "Untitled Pipeline",
  nodes: [],
  edges: [],
  comments: [],
  groups: [],
});

let nodeCounter = 1;
function nextNodeId() {
  return `node_${nodeCounter++}`;
}
function nextCommentId() {
  return `comment_${nodeCounter++}`;
}

function buildBlockLookup(blocks: BlockSpec[]): Map<string, BlockSpec> {
  const map = new Map<string, BlockSpec>();
  for (const block of blocks) {
    map.set(block.key, block);
    for (const alias of block.aliases ?? []) {
      const key = String(alias).trim();
      if (!key || map.has(key)) continue;
      map.set(key, block);
    }
  }
  return map;
}

function cloneValue<T>(value: T): T {
  if (typeof structuredClone === "function") {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value)) as T;
}

function cloneParamSchema(paramSchema: BlockParamSpec[] | undefined): BlockParamSpec[] {
  return (paramSchema ?? []).map((param) => ({ ...param }));
}

export function usePipeline() {
  const [blocks, setBlocks] = useState<BlockSpec[]>([]);
  const [nodes, setNodes] = useState<Node<ForgeNodeData>[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [pipelineId, setPipelineId] = useState<string | null>(null);
  const [pipelineFilename, setPipelineFilename] = useState(DEFAULT_PIPELINE_FILENAME);
  const [pipelineName, setPipelineName] = useState("Untitled Pipeline");
  const [groups, setGroups] = useState<PipelineGroup[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [isDirty, setIsDirty] = useState(false);

  // Keep node state separate so we can update it without React Flow re-rendering the whole graph
  const nodeStateMapRef = useRef<Map<string, NodeState>>(new Map());
  const lastSavedSignatureRef = useRef<string>(INITIAL_PIPELINE_SIGNATURE);
  const currentSignatureRef = useRef<string>(INITIAL_PIPELINE_SIGNATURE);
  const lastSavedFilenameRef = useRef<string>(DEFAULT_PIPELINE_FILENAME);
  const saveInFlightRef = useRef<Promise<string> | null>(null);
  const saveInFlightKeyRef = useRef<string | null>(null);
  const activeRunPipelineIdRef = useRef<string | null>(null);

  // ── Block registry ──────────────────────────────────────────────────────────

  useEffect(() => {
    fetchBlocks()
      .then(setBlocks)
      .catch(() => console.error("Failed to load block registry"));
  }, []);

  const reloadBlocks = useCallback(() => {
    fetchBlocks()
      .then(setBlocks)
      .catch(() => console.error("Failed to reload block registry"));
  }, []);

  // Backfill block metadata for already-mounted nodes after registry loads.
  useEffect(() => {
    if (blocks.length === 0) return;
    const blockMap = buildBlockLookup(blocks);
    setNodes((current) =>
      current.map((node) => {
        const spec = blockMap.get(String(node.data.blockKey));
        if (!spec) return node;
        return {
          ...node,
          data: {
            ...node.data,
            blockName: spec.name,
            category: spec.category,
            description: spec.description ?? "",
            n_inputs: spec.n_inputs,
            inputLabels: spec.input_labels ?? [],
            outputLabels: spec.output_labels ?? ["output"],
            paramSchema: cloneParamSchema(spec.param_schema),
            paramDescriptions: { ...(spec.param_descriptions ?? {}) },
          },
        };
      }),
    );
  }, [blocks]);

  // Track unsaved graph changes against last persisted payload.
  useEffect(() => {
    const signature = payloadSignature(
      buildPipelinePayload(nodes, edges, pipelineName, groups),
    );
    currentSignatureRef.current = signature;
    setIsDirty(
      signature !== lastSavedSignatureRef.current ||
      pipelineFilename !== lastSavedFilenameRef.current,
    );
  }, [nodes, edges, pipelineName, groups, pipelineFilename]);

  // ── Node state helpers ──────────────────────────────────────────────────────

  const setNodeStatus = useCallback(
    (nodeId: string, patch: Partial<NodeState>) => {
      const prev = nodeStateMapRef.current.get(nodeId) ?? { status: "idle" };
      const next = { ...prev, ...patch };
      nodeStateMapRef.current.set(nodeId, next);
      setNodes((ns) =>
        ns.map((n) =>
          n.id === makeRfId(nodeId)
            ? { ...n, data: { ...n.data, nodeState: next } }
            : n,
        ),
      );
    },
    [],
  );

  const markRunningNodesStale = useCallback(() => {
    setNodes((ns) =>
      ns.map((n) => {
        if (n.type === "commentBlock") return n;
        const current = nodeStateMapRef.current.get(n.id) ?? { status: "idle" };
        if (current.status !== "running") return n;
        const next: NodeState = {
          ...current,
          status: "stale",
          progressCurrent: undefined,
          progressTotal: undefined,
          progressPercent: undefined,
          progressLabel: undefined,
          progressDone: false,
        };
        nodeStateMapRef.current.set(n.id, next);
        return { ...n, data: { ...n.data, nodeState: next } };
      }),
    );
  }, []);

  // ── WebSocket ───────────────────────────────────────────────────────────────

  const handleWsMessage = useCallback(
    (msg: WsMessage) => {
      if (msg.type === "run_status") {
        if (msg.status === "started") {
          setRunError(null);
        } else if (msg.status === "complete") {
          setIsRunning(false);
          setIsStopping(false);
          activeRunPipelineIdRef.current = null;
        } else if (msg.status === "cancelled") {
          setIsRunning(false);
          setIsStopping(false);
          activeRunPipelineIdRef.current = null;
          markRunningNodesStale();
          setRunError(msg.message ?? "Execution cancelled");
        } else if (msg.status === "error") {
          setIsRunning(false);
          setIsStopping(false);
          activeRunPipelineIdRef.current = null;
          markRunningNodesStale();
          setRunError(msg.message ?? "Execution failed");
        }
      } else if (msg.type === "node_status") {
        const status: NodeStatus =
          msg.status === "running"
            ? "running"
            : msg.status === "error"
              ? "error"
              : "complete";
        setNodeStatus(msg.node_id, {
          status,
          checkpointId: msg.checkpoint_id,
          mode: msg.mode,
          errorMessage: msg.message,
          progressCurrent:
            status === "complete" ? undefined : status === "error" ? undefined : 0,
          progressTotal: undefined,
          progressPercent:
            status === "complete" ? 1 : status === "error" ? undefined : undefined,
          progressLabel: undefined,
          progressDone: status === "complete",
        });
      } else if (msg.type === "node_progress") {
        setNodeStatus(msg.node_id, {
          status: "running",
          progressCurrent: msg.current,
          progressTotal: msg.total,
          progressPercent:
            typeof msg.percent === "number"
              ? Math.max(0, Math.min(1, msg.percent))
              : undefined,
          progressLabel: msg.label,
          progressDone: msg.done ?? false,
        });
      }
      // run_result: individual node results are already handled by node_status messages
    },
    [markRunningNodesStale, setNodeStatus],
  );

  const handleWsClose = useCallback(() => {
    setIsRunning(false);
    setIsStopping(false);
    activeRunPipelineIdRef.current = null;
  }, []);

  const { connect: wsConnect, disconnect: wsDisconnect } = useWebSocket({
    onMessage: handleWsMessage,
    onClose: handleWsClose,
  });

  // ── Drop a block onto the canvas ────────────────────────────────────────────

  const addNode = useCallback(
    (spec: BlockSpec, position: { x: number; y: number }) => {
      const id = nextNodeId();
      const nodeState: NodeState = { status: "idle" };
      nodeStateMapRef.current.set(id, nodeState);
      const newNode: Node<ForgeNodeData> = {
        id,
        type: "forgeBlock",
        position,
        data: {
          blockKey: spec.key,
          blockName: spec.name,
          category: spec.category,
          description: spec.description ?? "",
          n_inputs: spec.n_inputs,
          inputLabels: spec.input_labels ?? [],
          outputLabels: spec.output_labels ?? ["output"],
          params: { ...spec.params },
          paramSchema: cloneParamSchema(spec.param_schema),
          paramDescriptions: { ...(spec.param_descriptions ?? {}) },
          notes: null,
          groupIds: [],
          nodeState,
        },
      };
      setNodes((ns) => [...ns, newNode]);
      return id;
    },
    [],
  );

  // ── Drop a comment annotation onto the canvas ────────────────────────────────

  const addComment = useCallback(
    (position: { x: number; y: number }) => {
      const id = nextCommentId();
      const newNode = {
        id,
        type: "commentBlock",
        position,
        zIndex: -1,
        style: { width: 320, height: 160 },
        data: {
          title: "",
          description: "",
          color: DEFAULT_COMMENT_COLOR,
          managed: false,
          groupId: id,
        } as CommentNodeData,
        connectable: false,
      };
      setNodes((ns) => [newNode as unknown as Node<ForgeNodeData>, ...ns]);
      return id;
    },
    [],
  );

  // ── Update params for a node ────────────────────────────────────────────────

  const updateNodeParams = useCallback(
    (nodeId: string, params: Record<string, unknown>) => {
      setNodes((ns) =>
        ns.map((n) =>
          n.id === nodeId
            ? { ...n, data: { ...n.data, params } }
            : n,
        ),
      );
      // Mark node and all descendants stale
      setNodes((ns) => {
        const descendants = getDescendants(nodeId, ns, edges);
        return ns.map((n) => {
          if (n.id === nodeId || descendants.has(n.id)) {
            const s: NodeState = { status: "stale" };
            nodeStateMapRef.current.set(n.id, s);
            return { ...n, data: { ...n.data, nodeState: s } };
          }
          return n;
        });
      });
    },
    [edges],
  );

  // ── Delete a node ───────────────────────────────────────────────────────────

  const deleteNode = useCallback((nodeId: string) => {
    nodeStateMapRef.current.delete(nodeId);
    setNodes((ns) => ns.filter((n) => n.id !== nodeId));
    setEdges((es) =>
      es.filter((e) => e.source !== nodeId && e.target !== nodeId),
    );
    setSelectedNodeId((prev) => (prev === nodeId ? null : prev));
  }, []);

  // ── Paste pre-copied nodes with new IDs ───────────────────────────────────

  const pasteNodes = useCallback(
    (
      sourceNodes: Node<ForgeNodeData>[],
      offset: { x: number; y: number } = { x: 40, y: 40 },
    ) => {
      if (sourceNodes.length === 0) return [];

      const pasted = sourceNodes.map((sourceNode) => {
        const isComment = sourceNode.type === "commentBlock";
        const newId = isComment ? nextCommentId() : nextNodeId();
        const clonedData = cloneValue(sourceNode.data);

        if (!isComment) {
          const copiedState =
            (clonedData as ForgeNodeData).nodeState ?? ({ status: "idle" } as NodeState);
          nodeStateMapRef.current.set(newId, copiedState);
          (clonedData as ForgeNodeData).nodeState = copiedState;
        }

        return {
          ...sourceNode,
          id: newId,
          position: {
            x: sourceNode.position.x + offset.x,
            y: sourceNode.position.y + offset.y,
          },
          data: clonedData as ForgeNodeData,
          selected: false,
          dragging: false,
        } as Node<ForgeNodeData>;
      });

      setNodes((ns) => [...ns, ...pasted]);
      return pasted.map((n) => n.id);
    },
    [],
  );

  const normalizePipelineFilenameDraft = useCallback(() => {
    setPipelineFilename((current) => slugifyPipelineFilename(current));
  }, []);

  // ── Persist pipeline ─────────────────────────────────────────────────────────

  const persistCurrentPipeline = useCallback(async () => {
    const payload = buildPipelinePayload(nodes, edges, pipelineName, groups);
    const signature = payloadSignature(payload);
    const requestedPipelineId = slugifyPipelineFilename(
      pipelineFilename || pipelineId || pipelineName || DEFAULT_PIPELINE_FILENAME,
    );
    const saveKey = `${signature}::${requestedPipelineId}`;

    if (
      saveInFlightRef.current &&
      saveInFlightKeyRef.current === saveKey
    ) {
      return await saveInFlightRef.current;
    }
    if (saveInFlightRef.current) {
      await saveInFlightRef.current;
    }

    const savePromise = (async () => {
      if (pipelineId && requestedPipelineId === pipelineId) {
        const env = await updatePipeline(pipelineId, payload);
        setPipelineId(env.id);
        setPipelineFilename(env.id);
        lastSavedSignatureRef.current = signature;
        lastSavedFilenameRef.current = env.id;
        setIsDirty(
          currentSignatureRef.current !== signature ||
          requestedPipelineId !== env.id,
        );
        return env.id;
      }
      if (pipelineId && requestedPipelineId !== pipelineId) {
        const env = await createPipeline(payload, requestedPipelineId);
        try {
          await apiDeletePipeline(pipelineId);
        } catch (error: unknown) {
          console.warn("Renamed pipeline but failed to delete previous file", error);
        }
        setPipelineId(env.id);
        setPipelineFilename(env.id);
        lastSavedSignatureRef.current = signature;
        lastSavedFilenameRef.current = env.id;
        setIsDirty(
          currentSignatureRef.current !== signature ||
          requestedPipelineId !== env.id,
        );
        return env.id;
      }
      const env = await createPipeline(payload, requestedPipelineId);
      setPipelineId(env.id);
      setPipelineFilename(env.id);
      lastSavedSignatureRef.current = signature;
      lastSavedFilenameRef.current = env.id;
      setIsDirty(
        currentSignatureRef.current !== signature ||
        requestedPipelineId !== env.id,
      );
      return env.id;
    })();

    saveInFlightRef.current = savePromise;
    saveInFlightKeyRef.current = saveKey;
    try {
      return await savePromise;
    } finally {
      if (saveInFlightRef.current === savePromise) {
        saveInFlightRef.current = null;
        saveInFlightKeyRef.current = null;
      }
    }
  }, [nodes, edges, pipelineName, pipelineId, pipelineFilename, groups]);

  // ── Run pipeline via WebSocket ──────────────────────────────────────────────

  const runPipeline = useCallback(async () => {
    if (isRunning) return;
    setIsRunning(true);
    setIsStopping(false);
    setRunError(null);

    try {
      const savedPipelineId = await persistCurrentPipeline();
      // Mark all nodes as idle before run so status is accurate
      setNodes((ns) =>
        ns.map((n) => {
          if (n.type === "commentBlock") return n;
          const s: NodeState = { status: "idle" };
          nodeStateMapRef.current.set(n.id, s);
          return { ...n, data: { ...n.data, nodeState: s } };
        }),
      );
      activeRunPipelineIdRef.current = savedPipelineId;
      wsConnect(savedPipelineId);
    } catch (error: unknown) {
      setIsRunning(false);
      setIsStopping(false);
      activeRunPipelineIdRef.current = null;
      setRunError(
        `Save before run failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }, [isRunning, persistCurrentPipeline, wsConnect]);

  const stopPipeline = useCallback(async () => {
    if (!isRunning) return;
    const runningPipelineId = activeRunPipelineIdRef.current ?? pipelineId;
    if (!runningPipelineId) return;

    setIsStopping(true);
    try {
      const result = await cancelPipelineExecution(runningPipelineId);
      if (result.status === "not_running") {
        setIsRunning(false);
        setIsStopping(false);
        activeRunPipelineIdRef.current = null;
      }
    } catch (error: unknown) {
      setIsStopping(false);
      setRunError(
        `Stop failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }, [isRunning, pipelineId]);

  // ── Save pipeline ───────────────────────────────────────────────────────────

  const savePipeline = useCallback(async () => {
    return await persistCurrentPipeline();
  }, [persistCurrentPipeline]);

  // ── Load pipeline ───────────────────────────────────────────────────────────

  const loadPipeline = useCallback(
    async (id: string) => {
      const env = await getPipeline(id);
      const { pipeline } = env;
      setPipelineId(env.id);
      setPipelineFilename(env.id);
      setPipelineName(pipeline.name);
      setGroups(pipeline.groups ?? []);

      // Fetch staleness
      let staleMap: Record<string, boolean> = {};
      try {
        const sr = await getStaleness(id);
        staleMap = sr.stale;
      } catch {
        // ignore — will show idle
      }

      nodeStateMapRef.current.clear();
      // Re-build nodeCounter to avoid collisions
      let maxCounter = 0;
      for (const n of pipeline.nodes) {
        const m = n.id.match(/node_(\d+)/);
        if (m) maxCounter = Math.max(maxCounter, parseInt(m[1], 10));
      }
      for (const c of pipeline.comments ?? []) {
        const m = c.id.match(/comment_(\d+)/);
        if (m) maxCounter = Math.max(maxCounter, parseInt(m[1], 10));
      }
      nodeCounter = maxCounter + 1;

      const blockMap = buildBlockLookup(blocks);

      const rfNodes: Node<ForgeNodeData>[] = pipeline.nodes.map(
        (pn: PipelineNode, idx: number) => {
          const spec = blockMap.get(pn.block);
          const hasSavedWidth = typeof pn.width === "number";
          const hasSavedHeight = typeof pn.height === "number";
          const nodeStyle =
            hasSavedWidth || hasSavedHeight
              ? {
                  ...(hasSavedWidth ? { width: pn.width } : {}),
                  ...(hasSavedHeight ? { height: pn.height } : {}),
                }
              : undefined;
          const isStale = staleMap[pn.id] ?? false;
          const nodeState: NodeState = {
            status: isStale ? "stale" : "idle",
          };
          nodeStateMapRef.current.set(pn.id, nodeState);
          return {
            id: pn.id,
            type: "forgeBlock",
            position: pn.position ?? { x: 100 + idx * 200, y: 200 },
            ...(nodeStyle ? { style: nodeStyle } : {}),
            data: {
              blockKey: pn.block,
              blockName: spec?.name ?? pn.block,
              category: spec?.category ?? "",
              description: spec?.description ?? "",
              n_inputs: spec?.n_inputs ?? 1,
              inputLabels: spec?.input_labels ?? [],
              outputLabels: spec?.output_labels ?? ["output"],
              params: { ...pn.params },
              paramSchema: cloneParamSchema(spec?.param_schema),
              paramDescriptions: { ...(spec?.param_descriptions ?? {}) },
              notes: pn.notes ?? null,
              groupIds: [...(pn.group_ids ?? [])],
              nodeState,
            },
          };
        },
      );

      const rfEdges: Edge[] = pipeline.edges.map((e, idx) => ({
        id: e.id ?? `e_${idx}`,
        source: e.source,
        target: e.target,
        sourceHandle:
          e.sourceHandle ??
          (typeof e.source_output === "number" ? `output_${e.source_output}` : undefined),
        targetHandle:
          e.targetHandle ??
          (typeof e.target_input === "number" ? `input_${e.target_input}` : undefined),
        type: "smoothstep",
        animated: false,
      }));

      const rfComments = (pipeline.comments ?? []).map((c) => ({
        id: c.id,
        type: "commentBlock",
        position: c.position,
        zIndex: -1,
        style: { width: c.width, height: c.height },
        data: {
          title: c.title,
          description: c.description,
          color: normalizeCommentColor(c.color) ?? DEFAULT_COMMENT_COLOR,
          managed: c.managed ?? false,
          groupId: c.group_id ?? c.id,
        } as CommentNodeData,
        connectable: false,
      }));

      const loadedNodes = [
        ...(rfComments as unknown as Node<ForgeNodeData>[]),
        ...rfNodes,
      ];
      const loadedSignature = payloadSignature(
        buildPipelinePayload(
          loadedNodes,
          rfEdges,
          pipeline.name,
          pipeline.groups ?? [],
        ),
      );
      lastSavedSignatureRef.current = loadedSignature;
      currentSignatureRef.current = loadedSignature;
      lastSavedFilenameRef.current = env.id;
      setIsDirty(false);

      setNodes(loadedNodes);
      setEdges(rfEdges);
      setSelectedNodeId(null);
    },
    [blocks],
  );

  const prettifyPipeline = useCallback(async () => {
    const savedPipelineId = await persistCurrentPipeline();
    await apiPrettifyPipeline(savedPipelineId);
    await loadPipeline(savedPipelineId);
  }, [loadPipeline, persistCurrentPipeline]);

  // ── New in-memory draft ─────────────────────────────────────────────────────

  const newPipelineDraft = useCallback(() => {
    nodeStateMapRef.current.clear();
    nodeCounter = 1;
    setNodes([]);
    setEdges([]);
    setPipelineId(null);
    setPipelineFilename(DEFAULT_PIPELINE_FILENAME);
    setPipelineName("Untitled Pipeline");
    setGroups([]);
    setSelectedNodeId(null);
    setIsRunning(false);
    setIsStopping(false);
    setRunError(null);
    activeRunPipelineIdRef.current = null;
    lastSavedSignatureRef.current = INITIAL_PIPELINE_SIGNATURE;
    currentSignatureRef.current = INITIAL_PIPELINE_SIGNATURE;
    lastSavedFilenameRef.current = DEFAULT_PIPELINE_FILENAME;
    setIsDirty(false);
  }, []);

  // ── Refresh staleness from backend ──────────────────────────────────────────

  const refreshStaleness = useCallback(async () => {
    if (!pipelineId) return;
    const sr = await getStaleness(pipelineId);
    setNodes((ns) =>
      ns.map((n) => {
        if (n.type === "commentBlock") return n;
        const isStale = sr.stale[n.id] ?? false;
        const curr = nodeStateMapRef.current.get(n.id) ?? { status: "idle" };
        if (curr.status === "running") return n; // don't interrupt running
        const next: NodeState = {
          ...curr,
          status: isStale ? "stale" : curr.status,
        };
        nodeStateMapRef.current.set(n.id, next);
        return { ...n, data: { ...n.data, nodeState: next } };
      }),
    );
  }, [pipelineId]);

  // ── Autosave ────────────────────────────────────────────────────────────────

  useEffect(() => {
    const autoSaveIntervalMs = 5 * 60 * 1000;
    const timer = window.setInterval(() => {
      if (isRunning || !isDirty) return;
      void persistCurrentPipeline().catch((error: unknown) => {
        console.error("Autosave failed", error);
      });
    }, autoSaveIntervalMs);
    return () => window.clearInterval(timer);
  }, [isRunning, isDirty, persistCurrentPipeline]);

  // Cleanup ws on unmount
  useEffect(() => {
    return () => wsDisconnect();
  }, [wsDisconnect]);

  return {
    // State
    blocks,
    nodes,
    setNodes,
    edges,
    setEdges,
    pipelineId,
    pipelineFilename,
    pipelineName,
    groups,
    setPipelineName,
    setPipelineFilename,
    normalizePipelineFilenameDraft,
    setGroups,
    selectedNodeId,
    setSelectedNodeId,
    isRunning,
    isStopping,
    isDirty,
    runError,
    // Actions
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
    refreshStaleness,
    reloadBlocks,
  };
}

// ── Utility ───────────────────────────────────────────────────────────────────

function getDescendants(
  nodeId: string,
  nodes: Node<ForgeNodeData>[],
  edges: Edge[],
): Set<string> {
  const all = new Set(nodes.map((n) => n.id));
  const adj = new Map<string, string[]>();
  for (const id of all) adj.set(id, []);
  for (const e of edges) {
    adj.get(e.source)?.push(e.target);
  }
  const visited = new Set<string>();
  const queue = [nodeId];
  while (queue.length) {
    const cur = queue.shift()!;
    for (const child of adj.get(cur) ?? []) {
      if (!visited.has(child)) {
        visited.add(child);
        queue.push(child);
      }
    }
  }
  return visited;
}
