/**
 * Full-screen setup overlay shown during first launch in Tauri mode.
 * Displays progress as the backend is being set up.
 */

import { useEffect, useState } from "react";
import type { SetupStage } from "@/hooks/useTauri";
import { FORGE_VERSION, resolveAppVersion } from "@/utils/version";

interface SetupScreenProps {
  stage: SetupStage;
  error: string | null;
  onRetry: () => void;
}

const STAGE_INFO: Record<
  Exclude<SetupStage, "ready">,
  { title: string; subtitle: string }
> = {
  checking: {
    title: "Detecting Python",
    subtitle: "Looking for Python 3.12+ on your system\u2026",
  },
  "creating-venv": {
    title: "Creating Environment",
    subtitle: "Setting up an isolated Python environment for Forge\u2026",
  },
  installing: {
    title: "Updating Dependencies",
    subtitle:
      "Installing data science packages. This may take a few minutes on first launch\u2026",
  },
  starting: {
    title: "Starting Forge",
    subtitle: "Starting the analysis engine\u2026",
  },
  "no-python": {
    title: "Python Not Found",
    subtitle:
      "Forge needs Python 3.12 or newer to run. Please install it, then click Retry.",
  },
  "file-locked": {
    title: "Files In Use",
    subtitle: "",
  },
  error: {
    title: "Setup Failed",
    subtitle: "Something went wrong while setting up Forge.",
  },
};

const STAGE_ORDER: SetupStage[] = [
  "checking",
  "creating-venv",
  "installing",
  "starting",
];

/** Detect if the error is a file-lock issue (another process holds venv files). */
function isFileLockError(error: string | null): boolean {
  if (!error) return false;
  return (
    error.includes("being used by another process") ||
    error.includes("WinError 32") ||
    error.includes("Permission denied") ||
    error.includes("EBUSY") ||
    error.includes("Resource busy")
  );
}

/** Rough OS detection for showing platform-appropriate help text. */
function detectOS(): "windows" | "macos" | "other" {
  if (typeof navigator === "undefined") return "other";
  const p = navigator.platform.toLowerCase();
  if (p.startsWith("win")) return "windows";
  if (p.startsWith("mac") || p.includes("mac")) return "macos";
  return "other";
}

export function SetupScreen({ stage, error, onRetry }: SetupScreenProps) {
  const [logPath, setLogPath] = useState<string | null>(null);
  const [version, setVersion] = useState<string>(FORGE_VERSION);

  useEffect(() => {
    (async () => {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const path = await invoke<string>("get_log_path");
        setLogPath(path);
      } catch { /* not in Tauri or command not available */ }
    })();
  }, []);

  useEffect(() => {
    let cancelled = false;
    void resolveAppVersion().then((v) => {
      if (!cancelled) setVersion(v);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  if (stage === "ready") return null;

  // Override stage to file-locked if we detect that specific error
  const effectiveStage: SetupStage =
    stage === "error" && isFileLockError(error) ? "file-locked" : stage;

  const info = STAGE_INFO[effectiveStage];
  const currentIndex = STAGE_ORDER.indexOf(effectiveStage);
  const isError =
    effectiveStage === "error" ||
    effectiveStage === "no-python" ||
    effectiveStage === "file-locked";

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-[#0f1117]">
      <div className="flex flex-col items-center gap-8 max-w-lg text-center px-6">
        {/* Logo / Title */}
        <div className="flex flex-col items-center gap-3">
          <img src="/forge-logo.png" alt="Forge" className="w-16 h-16" />
          <h1 className="text-3xl font-semibold text-white tracking-tight">
            Forge
          </h1>
          <p className="text-sm text-gray-500">
            Visual Data Pipeline Framework
          </p>
          <p className="text-xs text-gray-600 font-mono">v{version}</p>
        </div>

        {/* Progress steps */}
        {!isError && (
          <div className="flex items-center gap-2 w-full max-w-xs">
            {STAGE_ORDER.map((s, i) => (
              <div
                key={s}
                className={`h-1 flex-1 rounded-full transition-colors duration-500 ${
                  i < currentIndex
                    ? "bg-indigo-500"
                    : i === currentIndex
                      ? "bg-indigo-500 animate-pulse"
                      : "bg-gray-700"
                }`}
              />
            ))}
          </div>
        )}

        {/* Status */}
        <div className="flex flex-col items-center gap-2">
          <h2
            className={`text-lg font-medium ${isError ? "text-red-400" : "text-white"}`}
          >
            {info.title}
          </h2>
          <p className="text-sm text-gray-400">{info.subtitle}</p>
        </div>

        {/* File-locked help */}
        {effectiveStage === "file-locked" && (
          <div className="flex flex-col items-center gap-4">
            <p className="text-sm text-gray-300 leading-relaxed">
              Another program is using files that Forge needs to set up.
              This usually happens when Python is already running.
            </p>
            <div className="bg-gray-800/50 rounded-lg p-4 text-left w-full">
              {detectOS() === "macos" ? (
                <p className="text-xs text-gray-400 mb-2">
                  To fix this, close any running Python programs, then click
                  Retry. If you&apos;re not sure what&apos;s running, open{" "}
                  <span className="text-gray-300">Activity Monitor</span>{" "}
                  (press <span className="text-gray-300">Cmd + Space</span> and
                  search for &quot;Activity Monitor&quot;), look for{" "}
                  <span className="text-gray-300">Python</span> or{" "}
                  <span className="text-gray-300">python3</span>, and quit those
                  processes.
                </p>
              ) : (
                <p className="text-xs text-gray-400 mb-2">
                  To fix this, close any running Python programs, then click
                  Retry. If you&apos;re not sure what&apos;s running, open Task
                  Manager (<span className="text-gray-300">Ctrl + Shift + Esc</span>),
                  look for <span className="text-gray-300">python.exe</span>,
                  and end those tasks.
                </p>
              )}
            </div>
            <button
              onClick={onRetry}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors"
            >
              Retry Setup
            </button>
          </div>
        )}

        {/* Python not found */}
        {effectiveStage === "no-python" && (
          <div className="flex flex-col items-center gap-3">
            <a
              href="https://www.python.org/downloads/"
              target="_blank"
              rel="noopener noreferrer"
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors"
            >
              Download Python
            </a>
            <button
              onClick={onRetry}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-300 text-sm font-medium rounded-lg transition-colors"
            >
              Retry Detection
            </button>
          </div>
        )}

        {/* Generic error */}
        {effectiveStage === "error" && (
          <div className="flex flex-col items-center gap-3">
            {error && (
              <p className="text-xs text-red-400 font-mono bg-red-950/30 rounded px-3 py-2 max-w-full overflow-x-auto max-h-32 overflow-y-auto">
                {error}
              </p>
            )}
            {logPath && (
              <p className="text-xs text-gray-500">
                Full log: <span className="text-gray-400 font-mono select-all">{logPath}</span>
              </p>
            )}
            <button
              onClick={onRetry}
              className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors"
            >
              Retry Setup
            </button>
          </div>
        )}

        {/* Spinner for active stages */}
        {!isError && (
          <div className="w-6 h-6 border-2 border-gray-700 border-t-indigo-500 rounded-full animate-spin" />
        )}
      </div>
    </div>
  );
}
