import * as vscode from "vscode";
import { KernelClient } from "./kernel";

/** Extension host. Deliberately thin (DESIGN.md §3): registers the custom
 * editor and commands, bridges webview ⇄ kernel, watches .py files. All real
 * work happens in the kernel/worker processes; nothing heavy runs here. */

const kernels = new Map<string, KernelClient>();

function kernelFor(folder: vscode.WorkspaceFolder, output: vscode.OutputChannel): KernelClient {
  let kernel = kernels.get(folder.uri.toString());
  if (!kernel) {
    kernel = new KernelClient(folder, output);
    kernels.set(folder.uri.toString(), kernel);
  }
  return kernel;
}

export function activate(context: vscode.ExtensionContext): void {
  const output = vscode.window.createOutputChannel("*Forge");
  context.subscriptions.push(output);

  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(
      "starforge.canvas",
      new CanvasEditorProvider(context, output),
      { webviewOptions: { retainContextWhenHidden: true } },
    ),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("starforge.newPipeline", async () => {
      const folder = vscode.workspace.workspaceFolders?.[0];
      if (!folder) {
        vscode.window.showErrorMessage("*Forge needs an open folder to create a pipeline in.");
        return;
      }
      const name = await vscode.window.showInputBox({
        prompt: "Pipeline name",
        value: "pipeline",
        validateInput: (v) => (/^[\w.-]+$/.test(v) ? undefined : "Letters, digits, ., -, _ only"),
      });
      if (!name) return;
      const uri = vscode.Uri.joinPath(folder.uri, ".forge", "pipelines", `${name}.forge`);
      const doc = { schema: "starforge/1", name, nodes: [], edges: [] };
      await vscode.workspace.fs.writeFile(uri, Buffer.from(JSON.stringify(doc, null, 2) + "\n", "utf-8"));
      await vscode.commands.executeCommand("vscode.openWith", uri, "starforge.canvas");
    }),

    vscode.commands.registerCommand("starforge.rescan", async () => {
      const folder = vscode.workspace.workspaceFolders?.[0];
      if (folder) await kernelFor(folder, output).scan();
    }),

    vscode.commands.registerCommand("starforge.restartKernel", () => {
      for (const kernel of kernels.values()) kernel.shutdown();
      vscode.window.setStatusBarMessage("*Forge kernel restarted", 3000);
    }),
  );

  // Event-driven discovery (DESIGN.md §2.7): watcher events drive incremental
  // rescans; a reconcile sweep on window focus catches edits made while VS
  // Code was unfocused. No polling timers.
  const watcher = vscode.workspace.createFileSystemWatcher("**/*.py");
  context.subscriptions.push(watcher);
  let debounce: NodeJS.Timeout | undefined;
  const rescan = (folder: vscode.WorkspaceFolder) => {
    if (debounce) clearTimeout(debounce);
    debounce = setTimeout(() => kernelFor(folder, output).scan().catch(() => undefined), 250);
  };
  const onPyEvent = (uri: vscode.Uri) => {
    const folder = vscode.workspace.getWorkspaceFolder(uri);
    // Only rescan when a kernel already exists — never wake one for edits
    // in a workspace where no canvas has been opened.
    if (folder && kernels.has(folder.uri.toString())) rescan(folder);
  };
  context.subscriptions.push(
    watcher.onDidChange(onPyEvent),
    watcher.onDidCreate(onPyEvent),
    watcher.onDidDelete(onPyEvent),
    vscode.window.onDidChangeWindowState((state) => {
      if (!state.focused) return;
      for (const folder of vscode.workspace.workspaceFolders ?? []) {
        if (kernels.has(folder.uri.toString())) rescan(folder);
      }
    }),
  );
}

export function deactivate(): void {
  for (const kernel of kernels.values()) kernel.dispose();
  kernels.clear();
}

class CanvasEditorProvider implements vscode.CustomTextEditorProvider {
  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly output: vscode.OutputChannel,
  ) {}

  async resolveCustomTextEditor(
    document: vscode.TextDocument,
    panel: vscode.WebviewPanel,
    _token: vscode.CancellationToken,
  ): Promise<void> {
    const folder = vscode.workspace.getWorkspaceFolder(document.uri) ?? vscode.workspace.workspaceFolders?.[0];
    if (!folder) {
      void vscode.window.showErrorMessage("*Forge: open the containing folder to use the canvas.");
      return;
    }
    const kernel = kernelFor(folder, this.output);
    panel.webview.options = { enableScripts: true };
    panel.webview.html = this.html(panel.webview);

    const post = (type: string, payload: any) => void panel.webview.postMessage({ type, payload });

    const parseDoc = (): any | undefined => {
      try {
        return JSON.parse(document.getText() || "{}");
      } catch {
        return undefined; // mid-edit invalid JSON — keep last good state
      }
    };

    // Checkpoint artifacts (figure PNGs/HTML) live on disk under .forge/;
    // the webview can only load them through asWebviewUri.
    const artifactUri = (dir: string, file: string): string =>
      panel.webview.asWebviewUri(vscode.Uri.joinPath(folder.uri, dir, "outputs", file)).toString();

    const pushDoc = async () => {
      const doc = parseDoc();
      if (!doc) return;
      post("doc", doc);
      try {
        const hashes = await kernel.hashes(doc);
        post("hashes", hashes);
        await pushNodeFigures(doc, hashes);
      } catch (err: any) {
        this.output.appendLine(`[hashes] ${err.message}`);
      }
    };

    const pushNodeFigures = async (doc: any, hashes: Record<string, any>) => {
      const freshHashes = [
        ...new Set(
          Object.values(hashes)
            .filter((s: any) => s.history_hash && !s.stale)
            .map((s: any) => s.history_hash as string),
        ),
      ];
      const byNode: Record<string, { uri: string; kind: string }[]> = {};
      if (freshHashes.length > 0) {
        try {
          const { checkpoints } = await kernel.figures(freshHashes);
          for (const [nodeId, state] of Object.entries<any>(hashes)) {
            const checkpoint = state.history_hash ? checkpoints[state.history_hash] : undefined;
            if (checkpoint?.dir && checkpoint.artifacts.length > 0) {
              byNode[nodeId] = checkpoint.artifacts.map((a) => ({
                kind: a.kind,
                uri: artifactUri(checkpoint.dir!, a.file),
              }));
            }
          }
        } catch (err: any) {
          this.output.appendLine(`[figures] ${err.message}`);
        }
      }
      post("nodeFigures", byNode);
    };

    const subscriptions: vscode.Disposable[] = [];

    subscriptions.push(
      vscode.workspace.onDidChangeTextDocument((event) => {
        if (event.document.uri.toString() === document.uri.toString()) void pushDoc();
      }),
      kernel.onRunEvent.event((event) => {
        post("runEvent", event);
        // A finished run changes staleness; refresh node states immediately.
        if (event.event === "run_finished") void pushDoc();
      }),
      kernel.onPaletteChanged.event((palette) => {
        post("palette", palette);
        void pushDoc(); // source edits may change staleness
      }),
      panel.webview.onDidReceiveMessage(async (message) => {
        switch (message.type) {
          case "ready": {
            try {
              post("palette", await kernel.scan());
            } catch (err: any) {
              post("kernelError", err.message);
            }
            await pushDoc();
            break;
          }
          case "updateDoc": {
            const edit = new vscode.WorkspaceEdit();
            const text = JSON.stringify(message.payload, null, 2) + "\n";
            const fullRange = new vscode.Range(0, 0, document.lineCount, 0);
            edit.replace(document.uri, fullRange, text);
            await vscode.workspace.applyEdit(edit);
            break;
          }
          case "run": {
            const doc = parseDoc();
            if (!doc) break;
            try {
              post("runStarted", await kernel.runStart(doc));
            } catch (err: any) {
              post("kernelError", err.message);
            }
            break;
          }
          case "cancel":
            await kernel.runCancel(message.payload).catch(() => undefined);
            break;
          case "manifest": {
            try {
              const manifest = await kernel.manifest(message.payload);
              if (manifest?.dir) {
                for (const entry of manifest.outputs ?? []) {
                  if (entry.artifact?.file) entry.artifact.uri = artifactUri(manifest.dir, entry.artifact.file);
                }
                manifest.figures = (manifest.figures ?? []).map((f: any) => ({
                  ...f,
                  uri: artifactUri(manifest.dir, f.file),
                }));
              }
              post("manifest", manifest);
            } catch {
              post("manifest", null);
            }
            break;
          }
          case "openSource": {
            const uri = vscode.Uri.joinPath(folder.uri, message.payload.file);
            const source = await vscode.window.showTextDocument(uri, { viewColumn: vscode.ViewColumn.Beside });
            const line = Math.max(0, (message.payload.lineno ?? 1) - 1);
            source.revealRange(new vscode.Range(line, 0, line, 0), vscode.TextEditorRevealType.InCenter);
            break;
          }
        }
      }),
    );

    panel.onDidDispose(() => subscriptions.forEach((s) => s.dispose()));
  }

  private html(webview: vscode.Webview): string {
    const script = webview.asWebviewUri(
      vscode.Uri.joinPath(this.context.extensionUri, "media", "webview.js"),
    );
    const style = webview.asWebviewUri(
      vscode.Uri.joinPath(this.context.extensionUri, "media", "webview.css"),
    );
    const nonce = Math.random().toString(36).slice(2);
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}'; img-src ${webview.cspSource} data:; frame-src ${webview.cspSource};">
  <link rel="stylesheet" href="${style}">
</head>
<body>
  <div id="root"></div>
  <script nonce="${nonce}" src="${script}"></script>
</body>
</html>`;
  }
}
