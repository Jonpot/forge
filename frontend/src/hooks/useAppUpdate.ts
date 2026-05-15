import { useCallback, useEffect, useRef, useState } from "react";

export interface AvailableAppUpdate {
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

function detectTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

function buildConfirmationMessage(update: AvailableAppUpdate): string {
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

export function useAppUpdate() {
  const isTauri = detectTauri();
  const [availableUpdate, setAvailableUpdate] = useState<AvailableAppUpdate | null>(null);
  const [installing, setInstalling] = useState(false);
  const checkedRef = useRef(false);

  const checkForUpdate = useCallback(async () => {
    if (!isTauri) {
      return;
    }

    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const update = await invoke<AvailableAppUpdate | null>("check_app_update");
      setAvailableUpdate(update);
    } catch (error) {
      console.warn("Forge update check failed", error);
      setAvailableUpdate(null);
    }
  }, [isTauri]);

  useEffect(() => {
    if (!isTauri || checkedRef.current) {
      return;
    }
    checkedRef.current = true;
    void checkForUpdate();
  }, [checkForUpdate, isTauri]);

  const installUpdate = useCallback(async () => {
    if (!availableUpdate || installing) {
      return;
    }

    // Use the Tauri dialog plugin's `ask` rather than `window.confirm`:
    // WebView2 does not block JS execution on the native synchronous confirm,
    // so the installer would race ahead of the user's response.
    const { ask, message } = await import("@tauri-apps/plugin-dialog");
    const confirmed = await ask(buildConfirmationMessage(availableUpdate), {
      title: `Install Forge ${availableUpdate.version}?`,
      kind: "info",
      okLabel: "Install",
      cancelLabel: "Cancel",
    });
    if (!confirmed) {
      return;
    }

    setInstalling(true);
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const result = await invoke<InstallAppUpdateResult>("install_app_update");

      if (result.action === "auto-install") {
        const { getCurrentWindow } = await import("@tauri-apps/api/window");
        await getCurrentWindow().close();
        return;
      }

      if (result.action === "open-installer") {
        return;
      }

      // If we only had a release-page fallback, keep the button visible.
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      await message(`Forge update failed:\n\n${errorMessage}`, {
        title: "Update failed",
        kind: "error",
      });
    } finally {
      setInstalling(false);
    }
  }, [availableUpdate, installing]);

  return {
    availableUpdate,
    installing,
    installUpdate,
    checkForUpdate,
  };
}
