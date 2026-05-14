import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { readFileSync } from "fs";

// Tauri expects a fixed port during development
const FRONTEND_PORT = 40963;
const BACKEND_PORT = 40964;

const { version: FORGE_VERSION } = JSON.parse(
  readFileSync(path.resolve(__dirname, "package.json"), "utf-8"),
) as { version: string };

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  define: {
    __FORGE_VERSION__: JSON.stringify(FORGE_VERSION),
  },

  // Prevent Vite from obscuring Rust errors in the console
  clearScreen: false,

  server: {
    port: FRONTEND_PORT,
    // Tauri requires a strict port
    strictPort: true,
    proxy: {
      "/api": {
        target: `http://localhost:${BACKEND_PORT}`,
        changeOrigin: true,
      },
      "/api/ws": {
        target: `ws://localhost:${BACKEND_PORT}`,
        ws: true,
        changeOrigin: true,
      },
    },
  },

  // Env variables prefixed with TAURI_ are available in the frontend
  envPrefix: ["VITE_", "TAURI_"],

  build: {
    // Tauri uses Chromium on Windows and WebKit on macOS
    target: process.env.TAURI_ENV_PLATFORM === "windows" ? "chrome105" : "safari14",
    // Produce sourcemaps for debug builds
    sourcemap: !!process.env.TAURI_ENV_DEBUG,
    // Don't minify for debug builds
    minify: process.env.TAURI_ENV_DEBUG ? false : "esbuild",
  },
});
