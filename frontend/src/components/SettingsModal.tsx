import { useEffect, useMemo, useState } from "react";
import { getMcpConfig, type McpConfigResponse } from "@/api/client";
import {
  CATEGORY_COLOR_OPTIONS,
  CATEGORY_ICON_OPTIONS,
  categoryColorOption,
  getCategoryStyleOverrides,
  resolveCategoryStyleFromOverrides,
  saveCategoryStyleOverrides,
  serializeCategoryStyleOverrides,
  type CategoryStyleOverride,
} from "@/utils/categoryStyles";
import { FORGE_VERSION, resolveAppVersion } from "@/utils/version";

interface AvailableAppUpdate {
  currentVersion: string;
  version: string;
  publishedAt: string | null;
  releaseUrl: string;
  action: "auto-install" | "open-installer" | "open-release-page";
  assetName: string | null;
}

interface InstallAppUpdateResult {
  version: string;
  action: AvailableAppUpdate["action"];
  assetName: string | null;
}

function buildInstallConfirmation(update: AvailableAppUpdate): string {
  if (update.action === "auto-install") {
    return (
      `Download and install Forge ${update.version} now?\n\n` +
      "Forge will close, run the installer, and reopen when the update finishes.\n" +
      "Any unsaved changes will be lost."
    );
  }
  if (update.action === "open-installer") {
    return (
      `Download the Forge ${update.version} installer now?\n\n` +
      "Forge will open the downloaded installer when it finishes.\n" +
      "Any unsaved changes you have right now should be saved first."
    );
  }
  return (
    `Open the Forge ${update.version} release page now?\n\n` +
    "Any unsaved changes you have right now should be saved first."
  );
}

function detectTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
  customCategories: string[];
}

interface WorkspaceConfig {
  workspace_dir: string;
  setup_complete: boolean;
  auto_update_packages: boolean;
}

export function SettingsModal({
  open,
  onClose,
  customCategories,
}: SettingsModalProps) {
  const isTauri = detectTauri();
  const [config, setConfig] = useState<WorkspaceConfig | null>(null);
  const [workspaceDir, setWorkspaceDir] = useState("");
  const [autoUpdatePackages, setAutoUpdatePackages] = useState(false);
  const [categoryOverrides, setCategoryOverrides] = useState<
    Record<string, CategoryStyleOverride>
  >({});
  const [savedCategoryOverridesSnapshot, setSavedCategoryOverridesSnapshot] =
    useState("{}");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);
  const [updating, setUpdating] = useState(false);

  // App version + update state
  const [appVersion, setAppVersion] = useState<string>(FORGE_VERSION);
  const [checkingForUpdate, setCheckingForUpdate] = useState(false);
  const [updateCheckResult, setUpdateCheckResult] = useState<
    | { kind: "idle" }
    | { kind: "up-to-date"; checkedAt: number }
    | { kind: "available"; update: AvailableAppUpdate }
    | { kind: "error"; message: string }
  >({ kind: "idle" });
  const [installingAppUpdate, setInstallingAppUpdate] = useState(false);

  // MCP config state
  const [mcpConfig, setMcpConfig] = useState<McpConfigResponse | null>(null);
  const [mcpLoading, setMcpLoading] = useState(false);
  const [mcpError, setMcpError] = useState<string | null>(null);
  const [promptCopied, setPromptCopied] = useState(false);
  const [rawCopied, setRawCopied] = useState(false);

  // Load settings when the modal opens
  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setSaveMsg(null);
    const savedOverrides = getCategoryStyleOverrides();
    setCategoryOverrides(savedOverrides);
    setSavedCategoryOverridesSnapshot(
      serializeCategoryStyleOverrides(savedOverrides),
    );

    if (isTauri) {
      import("@tauri-apps/api/core").then(({ invoke }) => {
        invoke<WorkspaceConfig>("load_settings")
          .then((cfg) => {
            setConfig(cfg);
            setWorkspaceDir(cfg.workspace_dir);
            setAutoUpdatePackages(cfg.auto_update_packages ?? false);
          })
          .catch(() => {
            setConfig(null);
            setWorkspaceDir("");
            setAutoUpdatePackages(false);
          })
          .finally(() => setLoading(false));
      });
    } else {
      // In browser mode, nothing to load
      setLoading(false);
    }
  }, [open, isTauri]);

  // Load MCP config when the modal opens
  useEffect(() => {
    if (!open) return;
    setMcpLoading(true);
    setMcpError(null);
    getMcpConfig()
      .then((cfg) => setMcpConfig(cfg))
      .catch(() => setMcpError("Could not load MCP config. Is the backend running?"))
      .finally(() => setMcpLoading(false));
  }, [open]);

  // Resolve the running app version when the modal opens
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    void resolveAppVersion().then((v) => {
      if (!cancelled) setAppVersion(v);
    });
    return () => {
      cancelled = true;
    };
  }, [open]);

  // Reset transient update-check state each time the modal closes
  useEffect(() => {
    if (open) return;
    setUpdateCheckResult({ kind: "idle" });
    setCheckingForUpdate(false);
    setInstallingAppUpdate(false);
  }, [open]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  const sortedCustomCategories = useMemo(
    () =>
      [...new Set(customCategories)].sort((a, b) =>
        a.localeCompare(b, undefined, { sensitivity: "base" }),
      ),
    [customCategories],
  );

  if (!open) return null;

  const handleBrowse = async () => {
    try {
      const { open: openDialog } = await import(
        "@tauri-apps/plugin-dialog"
      );
      const selected = await openDialog({
        directory: true,
        multiple: false,
        title: "Select Workspace Directory",
        defaultPath: workspaceDir || undefined,
      });
      if (selected && typeof selected === "string") {
        setWorkspaceDir(selected);
      }
    } catch (err) {
      console.error("Failed to open folder dialog:", err);
    }
  };

  const handleSave = async () => {
    setSaving(true);
    setSaveMsg(null);
    try {
      if (isTauri) {
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("save_settings", {
          config: {
            workspace_dir: workspaceDir,
            setup_complete: true,
            auto_update_packages: autoUpdatePackages,
          },
        });
        setConfig({
          workspace_dir: workspaceDir,
          setup_complete: true,
          auto_update_packages: autoUpdatePackages,
        });
      }
      saveCategoryStyleOverrides(categoryOverrides);
      setSavedCategoryOverridesSnapshot(
        serializeCategoryStyleOverrides(categoryOverrides),
      );
      setSaveMsg("Settings saved");
      setTimeout(() => setSaveMsg(null), 3000);
    } catch (err) {
      setSaveMsg(
        "Save failed: " + (err instanceof Error ? err.message : String(err)),
      );
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    // Reset to original values
    if (config) {
      setWorkspaceDir(config.workspace_dir);
      setAutoUpdatePackages(config.auto_update_packages ?? false);
    }
    setCategoryOverrides(getCategoryStyleOverrides());
    setSavedCategoryOverridesSnapshot(
      serializeCategoryStyleOverrides(getCategoryStyleOverrides()),
    );
    setSaveMsg(null);
    onClose();
  };

  const handleCopyPrompt = async () => {
    if (!mcpConfig) return;
    try {
      await navigator.clipboard.writeText(mcpConfig.setup_prompt);
      setPromptCopied(true);
      setTimeout(() => setPromptCopied(false), 2500);
    } catch {
      // Fallback for environments where clipboard API is restricted
      const ta = document.createElement("textarea");
      ta.value = mcpConfig.setup_prompt;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      setPromptCopied(true);
      setTimeout(() => setPromptCopied(false), 2500);
    }
  };

  const handleCopyRawConfig = async () => {
    if (!mcpConfig) return;
    const raw = JSON.stringify(mcpConfig.config_json, null, 2);
    try {
      await navigator.clipboard.writeText(raw);
      setRawCopied(true);
      setTimeout(() => setRawCopied(false), 2500);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = raw;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
      setRawCopied(true);
      setTimeout(() => setRawCopied(false), 2500);
    }
  };

  const handleUpdatePackages = async () => {
    if (!isTauri) return;
    setUpdating(true);
    setSaveMsg(null);
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("update_packages");
      setSaveMsg("Packages updated successfully");
      setTimeout(() => setSaveMsg(null), 4000);
    } catch (err) {
      setSaveMsg(
        "Update failed: " + (err instanceof Error ? err.message : String(err)),
      );
    } finally {
      setUpdating(false);
    }
  };

  const handleCheckForAppUpdate = async () => {
    if (!isTauri || checkingForUpdate) return;
    setCheckingForUpdate(true);
    setUpdateCheckResult({ kind: "idle" });
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const result = await invoke<AvailableAppUpdate | null>("check_app_update");
      if (result) {
        setUpdateCheckResult({ kind: "available", update: result });
      } else {
        setUpdateCheckResult({ kind: "up-to-date", checkedAt: Date.now() });
      }
    } catch (err) {
      setUpdateCheckResult({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setCheckingForUpdate(false);
    }
  };

  const handleInstallAppUpdate = async (update: AvailableAppUpdate) => {
    if (!isTauri || installingAppUpdate) return;
    if (!window.confirm(buildInstallConfirmation(update))) return;
    setInstallingAppUpdate(true);
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const result = await invoke<InstallAppUpdateResult>("install_app_update");
      if (result.action === "auto-install") {
        const { getCurrentWindow } = await import("@tauri-apps/api/window");
        await getCurrentWindow().close();
        return;
      }
      // For open-installer / open-release-page, the OS now owns the next step.
    } catch (err) {
      setUpdateCheckResult({
        kind: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setInstallingAppUpdate(false);
    }
  };

  const hasChanges = config
    ? workspaceDir !== config.workspace_dir ||
      autoUpdatePackages !== (config.auto_update_packages ?? false) ||
      serializeCategoryStyleOverrides(categoryOverrides) !==
        savedCategoryOverridesSnapshot
    : serializeCategoryStyleOverrides(categoryOverrides) !==
        savedCategoryOverridesSnapshot;

  const updateCategoryOverride = (
    category: string,
    updater: (current: CategoryStyleOverride | undefined) => CategoryStyleOverride,
  ) => {
    setCategoryOverrides((current) => ({
      ...current,
      [category]: updater(current[category]),
    }));
  };

  const handleCategoryIconChange = (category: string, icon: string) => {
    updateCategoryOverride(category, (current) => ({
      ...current,
      icon,
    }));
  };

  const handleCategoryColorChange = (category: string, colorKey: string) => {
    updateCategoryOverride(category, (current) => ({
      ...current,
      colorKey,
    }));
  };

  const handleResetCategory = (category: string) => {
    setCategoryOverrides((current) => {
      const next = { ...current };
      delete next[category];
      return next;
    });
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Settings"
      className="fixed inset-0 z-40 flex items-start justify-center bg-black/60 px-4 py-4 backdrop-blur-sm animate-fade-in sm:px-6 sm:py-8"
      onClick={(e) => {
        if (e.target === e.currentTarget) handleCancel();
      }}
    >
      <div className="flex max-h-[calc(100vh-2rem)] w-full max-w-3xl flex-col overflow-hidden rounded-lg border border-forge-border bg-forge-surface shadow-2xl animate-fade-in-scale sm:max-h-[calc(100vh-4rem)]">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-forge-border">
          <h2 className="text-forge-text font-semibold text-sm">Settings</h2>
          <button
            onClick={handleCancel}
            aria-label="Close"
            className="text-forge-muted hover:text-forge-text transition-colors"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path
                d="M1 1l12 12M13 1L1 13"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-5 py-5 space-y-5">
          {loading ? (
            <p className="text-forge-muted text-sm text-center py-8 animate-pulse">
              Loading settings...
            </p>
          ) : (
            <>
              {isTauri ? (
                <>
                  {/* Workspace Directory */}
                  <div className="space-y-2">
                    <label className="block text-sm font-medium text-forge-text">
                      Workspace Directory
                    </label>
                    <p className="text-xs text-forge-muted">
                      Where Forge stores pipelines, datasets, and outputs.
                    </p>
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={workspaceDir}
                        onChange={(e) => setWorkspaceDir(e.target.value)}
                        autoCapitalize="off"
                        autoCorrect="off"
                        spellCheck={false}
                        className="flex-1 rounded border border-forge-border bg-forge-bg px-3 py-2 text-sm text-forge-text focus:border-forge-accent focus:outline-none font-mono"
                        placeholder="/path/to/workspace"
                      />
                      <button
                        onClick={() => { void handleBrowse(); }}
                        className="px-3 py-2 rounded text-sm text-forge-text bg-forge-border/40 hover:bg-forge-border/60 transition-colors flex-shrink-0"
                      >
                        Browse
                      </button>
                    </div>
                  </div>

                  {/* Info section */}
                  <div className="rounded border border-forge-border/50 bg-forge-bg/50 px-4 py-3 space-y-1">
                    <p className="text-xs text-forge-muted">
                      <span className="text-forge-text font-medium">Pipelines:</span>{" "}
                      {workspaceDir ? `${workspaceDir}/pipelines` : "---"}
                    </p>
                    <p className="text-xs text-forge-muted">
                      <span className="text-forge-text font-medium">Datasets:</span>{" "}
                      {workspaceDir ? `${workspaceDir}/datasets` : "---"}
                    </p>
                    <p className="text-xs text-forge-muted">
                      <span className="text-forge-text font-medium">Outputs:</span>{" "}
                      {workspaceDir ? `${workspaceDir}/outputs` : "---"}
                    </p>
                  </div>

                  {/* Python Dependency Updates */}
                  <div className="space-y-3">
                    <label className="block text-sm font-medium text-forge-text">
                      Python Dependencies
                    </label>
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-forge-text">Auto-update dependencies on boot</p>
                        <p className="text-xs text-forge-muted">
                          Forge itself updates automatically when the app build changes.
                          Enable this only if you also want third-party Python packages
                          like scipy or scikit-learn refreshed on every boot.
                        </p>
                      </div>
                      <button
                        role="switch"
                        aria-checked={autoUpdatePackages}
                        onClick={() => setAutoUpdatePackages((v) => !v)}
                        className={`ml-4 flex-shrink-0 w-8 h-4 rounded-full transition-colors flex items-center ${
                          autoUpdatePackages ? "bg-forge-accent" : "bg-forge-border"
                        }`}
                      >
                        <div
                          className={`w-3 h-3 rounded-full bg-white transition-transform ${
                            autoUpdatePackages ? "translate-x-4" : "translate-x-0.5"
                          }`}
                        />
                      </button>
                    </div>
                    <button
                      onClick={() => { void handleUpdatePackages(); }}
                      disabled={updating}
                      className="w-full px-3 py-2 rounded text-sm text-forge-text bg-forge-border/40 hover:bg-forge-border/60 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {updating ? "Updating dependencies..." : "Check for dependency updates"}
                    </button>
                  </div>
                </>
              ) : (
                <p className="text-forge-muted text-sm text-center py-2">
                  Desktop-only settings are unavailable in browser mode.
                </p>
              )}

              <div className="border-t border-forge-border/50" />

              <div className="space-y-3">
                <label className="block text-sm font-medium text-forge-text">
                  Custom Categories
                </label>
                <p className="text-xs text-forge-muted">
                  Pick the symbol and accent color Forge uses for plugin-defined categories.
                </p>
                {sortedCustomCategories.length === 0 ? (
                  <div className="rounded border border-forge-border/50 bg-forge-bg/50 px-4 py-3">
                    <p className="text-xs text-forge-muted">
                      No plugin-defined categories are loaded right now.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {sortedCustomCategories.map((category) => {
                      const preview = resolveCategoryStyleFromOverrides(
                        category,
                        categoryOverrides,
                      );
                      const selectedColor =
                        categoryColorOption(preview.colorKey) ??
                        CATEGORY_COLOR_OPTIONS[0];
                      const hasOverride = Boolean(
                        categoryOverrides[category]?.icon?.trim() ||
                          categoryOverrides[category]?.colorKey,
                      );

                      return (
                        <div
                          key={category}
                          className="rounded border border-forge-border/50 bg-forge-bg/50 px-4 py-3 space-y-3"
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="min-w-0">
                              <p className="text-sm font-medium text-forge-text">
                                {category}
                              </p>
                              <p className="text-[11px] text-forge-muted">
                                Preview
                              </p>
                            </div>
                            <div className="flex items-center gap-2">
                              <span
                                className={`inline-flex items-center gap-1 rounded px-2 py-1 text-[11px] font-medium ${preview.badgeClass}`}
                              >
                                <span aria-hidden="true">{preview.icon}</span>
                                {category}
                              </span>
                              <button
                                onClick={() => handleResetCategory(category)}
                                disabled={!hasOverride}
                                className="btn-ghost text-xs disabled:cursor-not-allowed disabled:opacity-50"
                              >
                                Reset
                              </button>
                            </div>
                          </div>

                          <div className="space-y-2">
                            <label className="block text-[11px] font-medium uppercase tracking-wider text-forge-muted">
                              Symbol
                            </label>
                            <div className="flex items-center gap-2">
                              <input
                                type="text"
                                value={categoryOverrides[category]?.icon ?? ""}
                                onChange={(e) =>
                                  handleCategoryIconChange(category, e.target.value)
                                }
                                maxLength={2}
                                className="w-14 rounded border border-forge-border bg-forge-surface px-2 py-1.5 text-center text-sm text-forge-text focus:border-forge-accent focus:outline-none"
                                aria-label={`${category} symbol`}
                              />
                              <div className="flex flex-wrap gap-1.5">
                                {CATEGORY_ICON_OPTIONS.map((icon) => (
                                  <button
                                    key={`${category}-${icon}`}
                                    onClick={() => handleCategoryIconChange(category, icon)}
                                    className={`h-8 w-8 rounded border text-sm transition-colors ${
                                      (categoryOverrides[category]?.icon ?? preview.icon) ===
                                      icon
                                        ? "border-forge-accent bg-forge-accent/15 text-forge-text"
                                        : "border-forge-border bg-forge-surface text-forge-muted hover:text-forge-text hover:border-forge-border-mid"
                                    }`}
                                    title={`Use ${icon}`}
                                  >
                                    {icon}
                                  </button>
                                ))}
                              </div>
                            </div>
                          </div>

                          <div className="space-y-2">
                            <label className="block text-[11px] font-medium uppercase tracking-wider text-forge-muted">
                              Color
                            </label>
                            <div className="flex flex-wrap gap-2">
                              {CATEGORY_COLOR_OPTIONS.map((option) => (
                                <button
                                  key={`${category}-${option.key}`}
                                  onClick={() =>
                                    handleCategoryColorChange(category, option.key)
                                  }
                                  className={`flex items-center gap-2 rounded border px-2 py-1.5 text-xs transition-colors ${
                                    selectedColor.key === option.key
                                      ? "border-forge-accent bg-forge-accent/10 text-forge-text"
                                      : "border-forge-border bg-forge-surface text-forge-muted hover:text-forge-text hover:border-forge-border-mid"
                                  }`}
                                  title={option.label}
                                >
                                  <span
                                    className={`inline-block h-3 w-3 rounded-full ${option.swatchClass}`}
                                  />
                                  {option.label}
                                </button>
                              ))}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </>
          )}

          {/* Divider before About section */}
          <div className="border-t border-forge-border/50" />

          {/* About / Updates */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-forge-text">
              About Forge
            </label>
            <div className="rounded border border-forge-border/50 bg-forge-bg/50 px-4 py-3 space-y-3">
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <p className="text-sm text-forge-text">
                    Forge{" "}
                    <span className="font-mono text-forge-muted">
                      v{appVersion}
                    </span>
                  </p>
                  <p className="text-xs text-forge-muted">
                    {isTauri
                      ? "Check GitHub Releases for newer Forge desktop builds."
                      : "Update checks are only available in the desktop app."}
                  </p>
                </div>
                {isTauri && (
                  <button
                    onClick={() => { void handleCheckForAppUpdate(); }}
                    disabled={checkingForUpdate || installingAppUpdate}
                    className="flex-shrink-0 px-3 py-1.5 rounded text-sm text-forge-text bg-forge-border/40 hover:bg-forge-border/60 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {checkingForUpdate ? "Checking..." : "Check for updates"}
                  </button>
                )}
              </div>

              {isTauri && updateCheckResult.kind === "up-to-date" && (
                <p className="text-xs text-forge-complete">
                  &#x2713; You&apos;re on the latest version.
                </p>
              )}

              {isTauri && updateCheckResult.kind === "available" && (
                <div className="space-y-2">
                  <p className="text-xs text-forge-text">
                    <span className="font-medium">
                      Forge {updateCheckResult.update.version}
                    </span>{" "}
                    is available.
                  </p>
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      onClick={() => {
                        void handleInstallAppUpdate(updateCheckResult.update);
                      }}
                      disabled={installingAppUpdate}
                      className="px-3 py-1.5 rounded text-sm font-medium text-white bg-forge-accent hover:bg-forge-accent-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {installingAppUpdate
                        ? "Installing..."
                        : updateCheckResult.update.action === "auto-install"
                          ? `Install ${updateCheckResult.update.version}`
                          : updateCheckResult.update.action === "open-installer"
                            ? "Download installer"
                            : "Open release page"}
                    </button>
                    <a
                      href={updateCheckResult.update.releaseUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-forge-muted hover:text-forge-text underline underline-offset-2"
                    >
                      Release notes
                    </a>
                  </div>
                </div>
              )}

              {isTauri && updateCheckResult.kind === "error" && (
                <p className="text-xs text-forge-error">
                  {updateCheckResult.message}
                </p>
              )}
            </div>
          </div>

          {/* Divider before MCP section */}
          <div className="border-t border-forge-border/50" />

          {/* AI Tools / MCP Setup */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-forge-text">
              AI Tools (MCP)
            </label>
            <p className="text-xs text-forge-muted">
              Connect Claude Code, Codex, or other AI tools to Forge's MCP server.
            </p>

            {mcpLoading ? (
              <p className="text-xs text-forge-muted animate-pulse py-1">
                Resolving paths...
              </p>
            ) : mcpError ? (
              <p className="text-xs text-forge-error py-1">{mcpError}</p>
            ) : (
              <div className="flex items-center gap-2 pt-1">
                <button
                  onClick={() => { void handleCopyPrompt(); }}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium text-white bg-forge-accent hover:bg-forge-accent-hover transition-colors"
                  title="Copy a self-contained prompt to paste into Claude Code or another AI tool — it will configure itself automatically."
                >
                  <svg width="13" height="13" viewBox="0 0 13 13" fill="none" className="flex-shrink-0">
                    <rect x="4" y="4" width="8" height="8" rx="1.5" stroke="currentColor" strokeWidth="1.25"/>
                    <path d="M9 4V2.5A1.5 1.5 0 0 0 7.5 1H2.5A1.5 1.5 0 0 0 1 2.5V7.5A1.5 1.5 0 0 0 2.5 9H4" stroke="currentColor" strokeWidth="1.25"/>
                  </svg>
                  {promptCopied ? "Copied!" : "Copy MCP Setup Prompt"}
                </button>
                <button
                  onClick={() => { void handleCopyRawConfig(); }}
                  className="px-3 py-1.5 rounded text-sm text-forge-text bg-forge-border/40 hover:bg-forge-border/60 transition-colors"
                  title="Copy just the raw JSON config block to paste manually."
                >
                  {rawCopied ? "Copied!" : "Copy Raw Config"}
                </button>
              </div>
            )}

            {/* Path preview */}
            {!mcpLoading && !mcpError && mcpConfig && (
              <div className="rounded border border-forge-border/50 bg-forge-bg/50 px-4 py-3 space-y-1 mt-2">
                <p className="text-xs text-forge-muted font-mono truncate" title={mcpConfig.python_executable}>
                  <span className="text-forge-text not-italic font-sans font-medium">Python:</span>{" "}
                  {mcpConfig.python_executable}
                </p>
                <p className="text-xs text-forge-muted font-mono truncate" title={mcpConfig.pipeline_dir}>
                  <span className="text-forge-text not-italic font-sans font-medium">Pipelines:</span>{" "}
                  {mcpConfig.pipeline_dir}
                </p>
                <p className="text-xs text-forge-muted font-mono truncate" title={mcpConfig.blocks_dir}>
                  <span className="text-forge-text not-italic font-sans font-medium">Blocks:</span>{" "}
                  {mcpConfig.blocks_dir}
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        {!loading && (
          <div className="flex items-center justify-between border-t border-forge-border px-5 py-3">
            <div className="min-h-[20px]">
              {saveMsg && (
                <span
                  className={`text-xs animate-fade-in-up ${
                    saveMsg.startsWith("Save failed") || saveMsg.startsWith("Update failed")
                      ? "text-forge-error"
                      : "text-forge-complete"
                  }`}
                >
                  {(saveMsg.startsWith("Settings saved") || saveMsg.startsWith("Packages updated")) ? "\u2713 " : ""}
                  {saveMsg}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button onClick={handleCancel} className="btn-ghost">
                Cancel
              </button>
              <button
                onClick={() => { void handleSave(); }}
                disabled={saving || !hasChanges}
                className="px-4 py-1.5 rounded text-sm font-semibold text-white bg-forge-accent hover:bg-forge-accent-hover transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {saving ? "Saving..." : "Save"}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
