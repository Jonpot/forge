import esbuild from "esbuild";

const production = process.argv.includes("--production");
const watch = process.argv.includes("--watch");

/** Extension host bundle (Node). */
const extensionCtx = await esbuild.context({
  entryPoints: ["src/extension.ts"],
  bundle: true,
  outfile: "dist/extension.js",
  platform: "node",
  format: "cjs",
  external: ["vscode"],
  sourcemap: !production,
  minify: production,
});

/** Webview bundle (browser iife) + css. */
const webviewCtx = await esbuild.context({
  entryPoints: ["src/webview/index.tsx"],
  bundle: true,
  outfile: "media/webview.js",
  platform: "browser",
  format: "iife",
  sourcemap: !production,
  minify: production,
  loader: { ".css": "css" },
});

if (watch) {
  await Promise.all([extensionCtx.watch(), webviewCtx.watch()]);
  console.log("[starforge] watching…");
} else {
  await Promise.all([extensionCtx.rebuild(), webviewCtx.rebuild()]);
  await Promise.all([extensionCtx.dispose(), webviewCtx.dispose()]);
  console.log("[starforge] build complete");
}
