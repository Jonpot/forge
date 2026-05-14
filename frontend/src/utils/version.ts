/**
 * Forge app version, injected at build time from frontend/package.json.
 * This stays in sync with pyproject.toml and src-tauri/Cargo.toml as part
 * of the release process.
 */
export const FORGE_VERSION: string = __FORGE_VERSION__;

function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

/**
 * Resolve the running app version. In Tauri this asks the runtime for the
 * value baked into the installed binary; in browser mode it falls back to
 * the build-time constant.
 */
export async function resolveAppVersion(): Promise<string> {
  if (isTauri()) {
    try {
      const { getVersion } = await import("@tauri-apps/api/app");
      return await getVersion();
    } catch {
      // Fall through to the build-time constant
    }
  }
  return FORGE_VERSION;
}
