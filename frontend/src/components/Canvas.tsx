import { useCallback, useRef, type MouseEvent } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  SelectionMode,
  type Connection,
  type Edge,
  type Node,
  type NodeTypes,
  type OnNodesChange,
  type OnEdgesChange,
  type ReactFlowInstance,
  BackgroundVariant,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { BlockNode } from "./BlockNode";
import { CommentNode } from "./CommentNode";
import type { CommentNodeData, ForgeNodeData } from "@/hooks/usePipeline";
import type { BlockSpec } from "@/types/pipeline";
import { buildCommentTheme } from "@/utils/commentColors";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const NODE_TYPES: NodeTypes = {
  forgeBlock: BlockNode as any,
  commentBlock: CommentNode as any,
};

interface CanvasProps {
  nodes: Node<ForgeNodeData>[];
  edges: Edge[];
  onNodesChange: OnNodesChange<Node<ForgeNodeData>>;
  onEdgesChange: OnEdgesChange;
  onConnect: (connection: Connection) => void;
  onNodeClick: (nodeId: string) => void;
  onPaneClick: () => void;
  onSelectionChange: (selectedNodes: Node<ForgeNodeData>[]) => void;
  onDropBlock: (spec: BlockSpec, position: { x: number; y: number }) => void;
  onDropComment: (position: { x: number; y: number }) => void;
  onDropBlockFile?: (file: File) => void;
  draggingSpec: BlockSpec | null;
  draggingComment: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onCanvasReady?: (instance: ReactFlowInstance<any, any>, wrapper: HTMLDivElement) => void;
}

export function Canvas({
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
  onConnect,
  onNodeClick,
  onPaneClick,
  onSelectionChange,
  onDropBlock,
  onDropComment,
  onDropBlockFile,
  draggingSpec,
  draggingComment,
  onCanvasReady,
}: CanvasProps) {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const rfInstance = useRef<ReactFlowInstance<any, any> | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    // Show a "copy" cursor for palette blocks; "link" for .py file installs
    const hasFiles = e.dataTransfer.types.includes("Files");
    e.dataTransfer.dropEffect = hasFiles ? "link" : "copy";
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (!reactFlowWrapper.current) return;

      // Handle .py file drops for block installation
      if (e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        if (file.name.endsWith(".py") && onDropBlockFile) {
          onDropBlockFile(file);
          return;
        }
      }

      if (!draggingSpec && !draggingComment) return;

      const bounds = reactFlowWrapper.current.getBoundingClientRect();
      const projectFn = rfInstance.current?.screenToFlowPosition;

      let position: { x: number; y: number };
      if (projectFn) {
        position = projectFn({ x: e.clientX, y: e.clientY });
      } else {
        position = {
          x: e.clientX - bounds.left - 85,
          y: e.clientY - bounds.top - 20,
        };
      }

      if (draggingComment) {
        onDropComment(position);
      } else if (draggingSpec) {
        onDropBlock(draggingSpec, position);
      }
    },
    [draggingSpec, draggingComment, onDropBlock, onDropComment, onDropBlockFile],
  );

  const centerViewport = useCallback((x: number, y: number) => {
    const instance = rfInstance.current;
    if (!instance) return;
    void instance.setCenter(x, y, {
      zoom: instance.getZoom(),
      duration: 220,
    });
  }, []);

  const handleMiniMapClick = useCallback(
    (event: MouseEvent<Element>, position: { x: number; y: number }) => {
      if (event.button !== 0) return;
      centerViewport(position.x, position.y);
    },
    [centerViewport],
  );

  const handleMiniMapNodeClick = useCallback(
    (event: MouseEvent<Element>, node: Node<ForgeNodeData>) => {
      if (event.button !== 0) return;
      const nodeWidth = typeof node.width === "number" ? node.width : 0;
      const nodeHeight = typeof node.height === "number" ? node.height : 0;
      centerViewport(
        node.position.x + nodeWidth / 2,
        node.position.y + nodeHeight / 2,
      );
    },
    [centerViewport],
  );

  return (
    <div
      ref={reactFlowWrapper}
      data-tour="canvas"
      className="flex-1 min-h-0 min-w-0 overflow-hidden bg-forge-bg relative"
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={NODE_TYPES}
        minZoom={0.1}
        maxZoom={2}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={(conn: Connection) => {
          onConnect(conn);
        }}
        onNodeClick={(_, node) => onNodeClick(node.id)}
        onPaneClick={onPaneClick}
        onSelectionChange={({ nodes: selectedNodes }) => {
          onSelectionChange(selectedNodes as Node<ForgeNodeData>[]);
        }}
        onInit={(instance) => {
          rfInstance.current = instance as unknown as ReactFlowInstance<any, any>;
          if (onCanvasReady && reactFlowWrapper.current) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            onCanvasReady(instance as unknown as ReactFlowInstance<any, any>, reactFlowWrapper.current);
          }
        }}
        fitView
        deleteKeyCode={["Delete", "Backspace"]}
        selectionOnDrag
        selectionMode={SelectionMode.Full}
        panOnDrag={[1, 2]}
        defaultEdgeOptions={{ type: "smoothstep" }}
        proOptions={{ hideAttribution: true }}
        style={{ background: "transparent" }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          color="#2a2d3a"
          gap={20}
          size={1}
        />
        <Controls
          className="!bg-forge-surface !border-forge-border !shadow-lg"
          showInteractive={false}
        />
        <MiniMap
          nodeColor={(n) => {
            if (n.type === "commentBlock") {
              const commentData = n.data as unknown as CommentNodeData;
              return buildCommentTheme(commentData.color).minimap;
            }
            const status = (n.data as ForgeNodeData)?.nodeState?.status ?? "idle";
            const colors: Record<string, string> = {
              idle: "#2a2d3a",
              stale: "#eab308",
              running: "#3b82f6",
              complete: "#22c55e",
              error: "#ef4444",
            };
            return colors[status] ?? "#2a2d3a";
          }}
          onClick={handleMiniMapClick}
          onNodeClick={handleMiniMapNodeClick}
          maskColor="rgba(15,17,23,0.7)"
          className="!bg-forge-surface !border !border-forge-border !rounded-lg !cursor-pointer"
        />
      </ReactFlow>

      {/* Empty state hint */}
      {nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center animate-fade-in-up max-w-xs">
            {/* Subtle forge icon */}
            <div className="mx-auto mb-4 w-12 h-12 rounded-xl bg-forge-surface border border-forge-border flex items-center justify-center animate-gentle-float">
              <img src="/forge-logo.png" alt="" aria-hidden="true" className="w-6 h-6" />
            </div>
            <p className="text-forge-text/70 text-sm font-medium">
              Drag blocks from the palette to get started
            </p>
            <p className="text-forge-muted/60 text-xs mt-2 leading-relaxed">
              Connect nodes by dragging between handles, then hit Run
            </p>
            <div className="mt-4 flex items-center justify-center gap-3 text-[10px] text-forge-muted/40">
              <span>Ctrl+C copy</span>
              <span className="w-px h-3 bg-forge-border/50" />
              <span>Ctrl+V paste</span>
              <span className="w-px h-3 bg-forge-border/50" />
              <span>Ctrl+Z undo</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Re-export addEdge so App can use it without importing from @xyflow/react directly
export { addEdge };
