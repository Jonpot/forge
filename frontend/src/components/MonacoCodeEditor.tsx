import type { CSSProperties } from "react";
import Editor, { loader } from "@monaco-editor/react";
import * as monaco from "monaco-editor";

// Bind @monaco-editor/react to the bundled monaco-editor (instead of fetching
// from the JSDelivr CDN). This makes the editor work offline / inside the
// packaged Tauri app. Run once at module load.
let configured = false;
function configureLoaderOnce() {
  if (configured) return;
  loader.config({ monaco });
  configured = true;
}

// Suppress the "Could not create web worker" warning from Monaco for
// languages without a configured worker (Python here). MonacoEnvironment is a
// global hook Monaco checks before spawning workers.
declare global {
  interface Window {
    MonacoEnvironment?: { getWorker?: (...args: unknown[]) => Worker };
  }
}
if (typeof window !== "undefined" && !window.MonacoEnvironment) {
  window.MonacoEnvironment = {
    getWorker: () => {
      // Return a no-op worker; Python only needs tokenization (no LSP).
      const code = "self.onmessage = () => {};";
      const blob = new Blob([code], { type: "application/javascript" });
      return new Worker(URL.createObjectURL(blob));
    },
  };
}

export interface MonacoCodeEditorProps {
  value: string;
  onChange: (next: string) => void;
  language?: string;
  height?: number | string;
  ariaLabel?: string;
  readOnly?: boolean;
  fontSize?: number;
}

export function MonacoCodeEditor({
  value,
  onChange,
  language = "python",
  height = 280,
  ariaLabel,
  readOnly = false,
  fontSize = 12,
}: MonacoCodeEditorProps) {
  configureLoaderOnce();
  const heightPx = typeof height === "number" ? `${height}px` : height;
  // When height is fixed numerically, expose vertical resize on the wrapper
  // (used in the side inspector). When height is fluid (e.g. "100%" inside a
  // NodeResizer-controlled BlockNode), let the parent dictate sizing.
  const wrapperStyle: CSSProperties = {
    height: heightPx,
    minHeight: 140,
    ...(typeof height === "number" ? { resize: "vertical" as const } : {}),
  };
  return (
    <div
      className="rounded border border-forge-border bg-forge-bg overflow-hidden"
      style={wrapperStyle}
      aria-label={ariaLabel}
    >
      <Editor
        height="100%"
        defaultLanguage={language}
        language={language}
        theme="vs-dark"
        value={value}
        onChange={(next) => onChange(next ?? "")}
        options={{
          minimap: { enabled: false },
          scrollBeyondLastLine: false,
          fontSize,
          lineHeight: Math.round(fontSize * 1.45),
          tabSize: 4,
          insertSpaces: true,
          wordWrap: "on",
          renderLineHighlight: "all",
          smoothScrolling: true,
          automaticLayout: true,
          fixedOverflowWidgets: true,
          readOnly,
          scrollbar: {
            verticalScrollbarSize: 10,
            horizontalScrollbarSize: 10,
          },
        }}
      />
    </div>
  );
}

export default MonacoCodeEditor;
