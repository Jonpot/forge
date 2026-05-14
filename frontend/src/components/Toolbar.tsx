import { useEffect, useRef, useState } from "react";
import {
  listPipelines,
  deletePipeline as apiDeletePipeline,
} from "@/api/client";
import type { PipelineSummary } from "@/types/pipeline";
import { SettingsModal } from "./SettingsModal";

interface ToolbarProps {
  pipelineName: string;
  pipelineId: string | null;
  pipelineFilename: string;
  customCategories: string[];
  appUpdate:
    | {
        version: string;
        action: "auto-install" | "open-installer" | "open-release-page";
        isInstalling: boolean;
      }
    | null;
  isRunning: boolean;
  isStopping: boolean;
  isDirty: boolean;
  runError: string | null;
  onNameChange: (name: string) => void;
  onFilenameChange: (name: string) => void;
  onFilenameBlur: () => void;
  onSave: () => Promise<string | undefined>;
  onPrettify: () => Promise<void>;
  onLoad: (id: string) => Promise<void>;
  onRun: () => void;
  onStop: () => Promise<void>;
  onNewPipeline: () => void;
  isExporting?: boolean;
  onExportPng?: () => void;
  onExportPdf?: () => void;
  onExportPython?: () => void;
  onExportNotebook?: () => void;
  onDownloadTemplate?: () => void;
  onInstallBlock?: () => void;
  onManagePlugins?: () => void;
  onInstallAppUpdate?: () => void;
}

export function Toolbar({
  pipelineName,
  pipelineId,
  pipelineFilename,
  customCategories,
  appUpdate,
  isRunning,
  isStopping,
  isDirty,
  runError,
  onNameChange,
  onFilenameChange,
  onFilenameBlur,
  onSave,
  onPrettify,
  onLoad,
  onRun,
  onStop,
  onNewPipeline,
  isExporting,
  onExportPng,
  onExportPdf,
  onExportPython,
  onExportNotebook,
  onDownloadTemplate,
  onInstallBlock,
  onManagePlugins,
  onInstallAppUpdate,
}: ToolbarProps) {
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const [showLoadPanel, setShowLoadPanel] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [pipelines, setPipelines] = useState<PipelineSummary[]>([]);
  const [loadingList, setLoadingList] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    setSaveMsg(null);
    try {
      await onSave();
      setSaveMsg("Saved");
    } catch (e: unknown) {
      setSaveMsg(
        "Save failed: " + (e instanceof Error ? e.message : String(e)),
      );
    } finally {
      setSaving(false);
      setTimeout(() => setSaveMsg(null), 3000);
    }
  };

  const openLoadPanel = async () => {
    setShowLoadPanel(true);
    setLoadingList(true);
    try {
      const list = await listPipelines();
      setPipelines(list);
    } catch {
      setPipelines([]);
    } finally {
      setLoadingList(false);
    }
  };

  const handleLoad = async (id: string) => {
    setShowLoadPanel(false);
    await onLoad(id);
  };

  const handleDelete = async (id: string) => {
    await apiDeletePipeline(id);
    setPipelines((ps) => ps.filter((p) => p.id !== id));
  };

  const updateTitle = appUpdate
    ? appUpdate.action === "auto-install"
      ? `Download and install Forge ${appUpdate.version}`
      : appUpdate.action === "open-installer"
        ? `Download the Forge ${appUpdate.version} installer`
        : `Open the Forge ${appUpdate.version} release page`
    : "";

  return (
    <>
      <header className="h-12 flex-shrink-0 bg-forge-surface border-b border-forge-border flex items-center gap-3 px-4">
        {/* App logo / title */}
        <div className="flex items-center gap-2 mr-2">
          <img src="/forge-logo.png" alt="Forge" className="w-5 h-5" />
          <span className="text-forge-accent font-bold text-base tracking-tight">
            Forge
          </span>
        </div>

        <div className="h-5 w-px bg-forge-border" />

        {/* Pipeline name */}
        <input
          className="bg-transparent text-forge-text text-sm font-medium border-b border-transparent hover:border-forge-border focus:border-forge-accent focus:outline-none transition-colors min-w-0"
          value={pipelineName}
          size={Math.max(pipelineName.length, 1)}
          onChange={(e) => onNameChange(e.target.value)}
          aria-label="Pipeline name"
          autoCapitalize="off"
          autoCorrect="off"
          spellCheck={false}
        />

        <input
          className="hidden sm:block min-w-0 bg-transparent text-forge-muted text-[11px] font-mono border-b border-transparent hover:border-forge-border focus:border-forge-accent focus:outline-none transition-colors"
          value={pipelineFilename}
          size={Math.max(pipelineFilename.length, 1)}
          onChange={(e) => onFilenameChange(e.target.value)}
          onBlur={onFilenameBlur}
          aria-label="Pipeline filename"
          autoCapitalize="off"
          autoCorrect="off"
          spellCheck={false}
        />

        <div className="flex-1" />

        {/* Status / error */}
        {runError && (
          <span className="text-forge-error text-xs max-w-xs truncate" title={runError}>
            ⚠ {runError}
          </span>
        )}
        {saveMsg && (
          <span
            className={`text-xs animate-fade-in-up ${saveMsg.startsWith("Save failed") ? "text-forge-error" : "text-forge-complete"}`}
          >
            {saveMsg === "Saved" ? "✓ " : ""}{saveMsg}
          </span>
        )}
        {isDirty && (
          <span className="text-forge-muted text-xs hidden md:inline">
            Unsaved changes
          </span>
        )}

        {/* Action buttons */}
        <button
          onClick={onNewPipeline}
          className="btn-ghost"
          title="New pipeline"
        >
          New
        </button>

        <button onClick={openLoadPanel} className="btn-ghost" title="Open pipeline">
          Open
        </button>

        <button
          onClick={handleSave}
          disabled={saving}
          className="btn-ghost"
          title="Save pipeline"
        >
          {saving ? "Saving…" : "Save"}
        </button>

        <button
          onClick={() => {
            void onPrettify();
          }}
          className="btn-ghost"
          title="Auto-layout blocks and sync groups from comments"
        >
          Prettify
        </button>

        <div className="h-5 w-px bg-forge-border" />

        {/* Export menu */}
        <div className="relative group">
          <button
            disabled={isExporting}
            className="btn-ghost inline-flex items-center gap-1"
            title="Export the current pipeline"
          >
            {isExporting ? (
              <span className="flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-full bg-forge-accent animate-pulse" />
                Exporting…
              </span>
            ) : (
              <>
                <span>Export</span>
                <span className="text-[10px]">▾</span>
              </>
            )}
          </button>

          <div
            className="
              absolute right-0 top-full z-50 min-w-44 pt-1
              opacity-0 pointer-events-none translate-y-1
              transition-all duration-150
              group-hover:opacity-100 group-hover:pointer-events-auto group-hover:translate-y-0
              group-focus-within:opacity-100 group-focus-within:pointer-events-auto group-focus-within:translate-y-0
            "
          >
            <div className="overflow-hidden rounded-md border border-forge-border bg-forge-surface shadow-2xl">
              <button
                onClick={onExportPng}
                disabled={isExporting}
                className="w-full px-3 py-2 text-left text-sm text-forge-text hover:bg-forge-bg/50 transition-colors disabled:text-forge-muted disabled:hover:bg-transparent"
                title="Export full-resolution PNG of the entire pipeline"
              >
                PNG
              </button>
              <button
                onClick={onExportPdf}
                disabled={isExporting}
                className="w-full px-3 py-2 text-left text-sm text-forge-text hover:bg-forge-bg/50 transition-colors disabled:text-forge-muted disabled:hover:bg-transparent"
                title="Export full-resolution PDF of the entire pipeline"
              >
                PDF
              </button>
              <button
                onClick={onExportPython}
                disabled={isExporting}
                className="w-full px-3 py-2 text-left text-sm text-forge-text hover:bg-forge-bg/50 transition-colors disabled:text-forge-muted disabled:hover:bg-transparent"
                title="Download a runnable Python export bundle"
              >
                Python Script
              </button>
              <button
                onClick={onExportNotebook}
                disabled={isExporting}
                className="w-full px-3 py-2 text-left text-sm text-forge-text hover:bg-forge-bg/50 transition-colors disabled:text-forge-muted disabled:hover:bg-transparent"
                title="Download a runnable Jupyter notebook export bundle"
              >
                Jupyter Notebook
              </button>
            </div>
          </div>
        </div>

        {/* Plugins menu */}
        <div className="relative group">
          <button
            className="btn-ghost inline-flex items-center gap-1"
            title="Manage custom block plugins"
          >
            <span>Plugins</span>
            <span className="text-[10px]">▾</span>
          </button>

          <div
            className="
              absolute right-0 top-full z-50 min-w-52 pt-1
              opacity-0 pointer-events-none translate-y-1
              transition-all duration-150
              group-hover:opacity-100 group-hover:pointer-events-auto group-hover:translate-y-0
              group-focus-within:opacity-100 group-focus-within:pointer-events-auto group-focus-within:translate-y-0
            "
          >
            <div className="overflow-hidden rounded-md border border-forge-border bg-forge-surface shadow-2xl">
              <div className="px-3 py-2 border-b border-forge-border">
                <p className="text-[10px] text-forge-muted font-medium uppercase tracking-wider">Block Plugins</p>
              </div>
              <button
                onClick={onManagePlugins}
                className="w-full px-3 py-2 text-left text-sm text-forge-text hover:bg-forge-bg/50 transition-colors flex items-center gap-2"
                title="Open the installed plugin manager"
              >
                <span aria-hidden="true" className="text-forge-muted">☰</span>
                Manage Plugins…
              </button>
              <button
                onClick={onDownloadTemplate}
                className="w-full px-3 py-2 text-left text-sm text-forge-text hover:bg-forge-bg/50 transition-colors flex items-center gap-2"
                title="Download a template .py file showing how to build a custom plugin"
              >
                <span aria-hidden="true" className="text-forge-muted">↓</span>
                Download Plugin Template
              </button>
              <button
                onClick={onInstallBlock}
                className="w-full px-3 py-2 text-left text-sm text-forge-text hover:bg-forge-bg/50 transition-colors flex items-center gap-2"
                title="Pick a .py file to install as a custom block plugin"
              >
                <span aria-hidden="true" className="text-forge-muted">+</span>
                Install Plugin from File…
              </button>
              <div className="px-3 py-2 border-t border-forge-border">
                <p className="text-[10px] text-forge-muted leading-relaxed">
                  Or drag a .py file onto the canvas to install.
                </p>
              </div>
            </div>
          </div>
        </div>

        {appUpdate && (
          <button
            onClick={onInstallAppUpdate}
            disabled={appUpdate.isInstalling}
            className={`
              inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm font-semibold
              transition-[background-color,border-color,color,transform] duration-150
              ${
                appUpdate.isInstalling
                  ? "cursor-progress border-[#6ee7b7]/30 bg-[#0f2f26] text-[#9cf3d0]"
                  : "border-[#6ee7b7]/35 bg-[#12372d] text-[#8bf5cf] hover:bg-[#184438] hover:border-[#86efcc]/60 active:scale-[0.97]"
              }
            `}
            title={updateTitle}
            aria-label={updateTitle}
          >
            {appUpdate.isInstalling ? (
              <span className="inline-block h-3.5 w-3.5 rounded-full border-2 border-[#9cf3d0]/40 border-t-[#9cf3d0] animate-spin" />
            ) : (
              <svg
                width="14"
                height="14"
                viewBox="0 0 16 16"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M8 2.5v7" />
                <path d="m5.4 7.8 2.6 2.7 2.6-2.7" />
                <path d="M3 12.5h10" />
              </svg>
            )}
            <span className="hidden lg:inline">
              {appUpdate.isInstalling ? "Updating…" : `v${appUpdate.version}`}
            </span>
          </button>
        )}

        {/* Settings gear */}
        <button
          onClick={() => setShowSettings(true)}
          className="btn-ghost p-1.5"
          title="Settings"
          aria-label="Settings"
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 20 20"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="10" cy="10" r="3" />
            <path d="M11.8 1.5h-3.6l-.5 2.4a6.5 6.5 0 0 0-1.7 1l-2.3-.8-1.8 3.1 1.8 1.6a6.5 6.5 0 0 0 0 2l-1.8 1.6 1.8 3.1 2.3-.8a6.5 6.5 0 0 0 1.7 1l.5 2.4h3.6l.5-2.4a6.5 6.5 0 0 0 1.7-1l2.3.8 1.8-3.1-1.8-1.6a6.5 6.5 0 0 0 0-2l1.8-1.6-1.8-3.1-2.3.8a6.5 6.5 0 0 0-1.7-1z" />
          </svg>
        </button>

        <div className="h-5 w-px bg-forge-border" />

        <div data-tour="run-button" className="flex items-center gap-2">
          <button
            onClick={onRun}
            disabled={isRunning}
            className={`
              px-4 py-1.5 rounded text-sm font-semibold whitespace-nowrap
              transition-[color,background-color,transform,box-shadow]
              duration-150
              ${
                isRunning
                  ? "bg-forge-running/30 text-forge-running cursor-wait"
                  : "bg-forge-accent hover:bg-forge-accent-hover hover:shadow-lg hover:shadow-forge-accent/20 active:scale-[0.96] text-white cursor-pointer"
              }
            `}
            title="Run pipeline (auto-saves first)"
          >
            {isRunning ? (
              <span className="flex items-center gap-2">
                <span className="inline-block w-2 h-2 rounded-full bg-forge-running animate-pulse" />
                Running…
              </span>
            ) : (
              "▶ Run"
            )}
          </button>
          {isRunning && (
            <button
              onClick={() => {
                void onStop();
              }}
              disabled={isStopping}
              className={`
                px-3 py-1.5 rounded text-sm font-semibold
                transition-[color,background-color,transform] duration-150
                ${
                  isStopping
                    ? "bg-forge-error/30 text-forge-error cursor-progress"
                    : "bg-forge-error hover:bg-forge-error/90 active:scale-[0.96] text-white cursor-pointer"
                }
              `}
              title="Hard stop the active pipeline run"
            >
              {isStopping ? "Stopping…" : "■ Stop"}
            </button>
          )}
        </div>
      </header>

      {/* Load panel */}
      {showLoadPanel && (
        <LoadPanel
          pipelines={pipelines}
          loading={loadingList}
          currentId={pipelineId}
          onLoad={handleLoad}
          onDelete={handleDelete}
          onClose={() => setShowLoadPanel(false)}
        />
      )}

      {/* Settings modal */}
      <SettingsModal
        open={showSettings}
        onClose={() => setShowSettings(false)}
        customCategories={customCategories}
      />
    </>
  );
}

// ── Load panel ────────────────────────────────────────────────────────────────

interface LoadPanelProps {
  pipelines: PipelineSummary[];
  loading: boolean;
  currentId: string | null;
  onLoad: (id: string) => void;
  onDelete: (id: string) => Promise<void>;
  onClose: () => void;
}

function LoadPanel({
  pipelines,
  loading,
  currentId,
  onLoad,
  onDelete,
  onClose,
}: LoadPanelProps) {
  const [pipelineToDelete, setPipelineToDelete] = useState<PipelineSummary | null>(
    null,
  );
  const [deleteConfirmationText, setDeleteConfirmationText] = useState("");
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const deleteInputRef = useRef<HTMLInputElement | null>(null);
  const deleteKeyword = "delete";

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key !== "Escape") {
        return;
      }
      if (pipelineToDelete) {
        if (!isDeleting) {
          setPipelineToDelete(null);
          setDeleteConfirmationText("");
          setDeleteError(null);
        }
        return;
      }
      onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isDeleting, onClose, pipelineToDelete]);

  useEffect(() => {
    if (pipelineToDelete) {
      deleteInputRef.current?.focus();
    }
  }, [pipelineToDelete]);

  const closeDeleteDialog = () => {
    if (isDeleting) {
      return;
    }
    setPipelineToDelete(null);
    setDeleteConfirmationText("");
    setDeleteError(null);
  };

  const handleDeleteRequest = (pipeline: PipelineSummary) => {
    setPipelineToDelete(pipeline);
    setDeleteConfirmationText("");
    setDeleteError(null);
  };

  const handleConfirmDelete = async () => {
    if (!pipelineToDelete) {
      return;
    }
    if (deleteConfirmationText.trim().toLowerCase() !== deleteKeyword) {
      return;
    }

    setIsDeleting(true);
    setDeleteError(null);
    try {
      await onDelete(pipelineToDelete.id);
      setPipelineToDelete(null);
      setDeleteConfirmationText("");
    } catch (error: unknown) {
      setDeleteError(
        error instanceof Error ? error.message : "Failed to delete pipeline.",
      );
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Open Pipeline"
      className="fixed inset-0 z-40 flex items-start justify-center pt-20 bg-black/60 backdrop-blur-sm animate-fade-in"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="w-full max-w-md bg-forge-surface border border-forge-border rounded-lg shadow-2xl overflow-hidden animate-fade-in-scale">
        <div className="flex items-center justify-between px-5 py-3 border-b border-forge-border">
          <h2 className="text-forge-text font-semibold text-sm">
            Open Pipeline
          </h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="text-forge-muted hover:text-forge-text transition-colors"
          >
            ✕
          </button>
        </div>

        <div className="max-h-96 overflow-y-auto">
          {loading ? (
            <p className="text-forge-muted text-xs text-center py-8 animate-pulse">
              Loading…
            </p>
          ) : pipelines.length === 0 ? (
            <p className="text-forge-muted text-xs text-center py-8">
              No saved pipelines
            </p>
          ) : (
            <ul className="divide-y divide-forge-border">
              {pipelines.map((p) => (
                <li
                  key={p.id}
                  className="flex items-center justify-between px-5 py-3 hover:bg-forge-bg/40 transition-colors"
                >
                  <div className="min-w-0">
                    <p
                      className={`text-sm font-medium truncate ${
                        p.id === currentId
                          ? "text-forge-accent"
                          : "text-forge-text"
                      }`}
                    >
                      {p.name}
                    </p>
                    <p className="text-[11px] text-forge-muted font-mono">
                      {p.id}
                    </p>
                    <p className="text-[11px] text-forge-muted">
                      {new Date(p.updated_at * 1000).toLocaleString()}
                    </p>
                  </div>
                  <div className="flex items-center gap-2 ml-4 flex-shrink-0">
                    <button
                      onClick={() => onLoad(p.id)}
                      className="text-xs text-forge-accent hover:text-forge-accent-hover transition-colors font-medium"
                    >
                      Open
                    </button>
                    <button
                      onClick={() => handleDeleteRequest(p)}
                      className="text-xs text-forge-muted hover:text-forge-error transition-colors"
                      title="Delete pipeline"
                      aria-label={`Delete pipeline ${p.name}`}
                    >
                      ✕
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {pipelineToDelete && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              closeDeleteDialog();
            }
          }}
        >
          <div className="w-full max-w-sm rounded-lg border border-forge-border bg-forge-surface shadow-2xl">
            <div className="border-b border-forge-border px-5 py-4">
              <h3 className="text-sm font-semibold text-forge-text">
                Delete Pipeline
              </h3>
            </div>

            <div className="space-y-3 px-5 py-4">
              <p className="text-sm text-forge-text">
                This permanently deletes the saved pipeline{" "}
                <span className="font-semibold">{pipelineToDelete.name}</span>.
              </p>
              <p className="text-xs text-forge-muted">
                Type <span className="font-mono text-forge-text">delete</span> to
                confirm.
              </p>
              <input
                ref={deleteInputRef}
                value={deleteConfirmationText}
                onChange={(e) => setDeleteConfirmationText(e.target.value)}
                className="w-full rounded border border-forge-border bg-forge-bg px-3 py-2 text-sm text-forge-text focus:border-forge-error focus:outline-none"
                placeholder={deleteKeyword}
                aria-label="Type delete to confirm"
                autoCapitalize="off"
                autoComplete="off"
                autoCorrect="off"
                spellCheck={false}
              />
              {deleteError && (
                <p className="text-xs text-forge-error">{deleteError}</p>
              )}
            </div>

            <div className="flex items-center justify-end gap-2 border-t border-forge-border px-5 py-4">
              <button
                onClick={closeDeleteDialog}
                disabled={isDeleting}
                className="btn-ghost disabled:cursor-not-allowed disabled:opacity-60"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  void handleConfirmDelete();
                }}
                disabled={
                  isDeleting ||
                  deleteConfirmationText.trim().toLowerCase() !== deleteKeyword
                }
                className="rounded bg-forge-error px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-forge-error/90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isDeleting ? "Deleting…" : "Delete pipeline"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
