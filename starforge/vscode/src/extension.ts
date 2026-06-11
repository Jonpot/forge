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

    vscode.commands.registerCommand("starforge.cleanCheckpoints", async () => {
      const folder = vscode.workspace.workspaceFolders?.[0];
      if (!folder) return;
      try {
        const stats = await kernelFor(folder, output).gcCheckpoints();
        const freed = (stats.freed_bytes / (1024 * 1024)).toFixed(1);
        const remaining = (stats.remaining_bytes / (1024 * 1024)).toFixed(1);
        void vscode.window.showInformationMessage(
          `*Forge: evicted ${stats.deleted} checkpoints (${freed} MB freed, ${remaining} MB in use).`,
        );
      } catch (err: any) {
        void vscode.window.showErrorMessage(`*Forge: clean failed — ${err.message}`);
      }
    }),

    vscode.commands.registerCommand(
      "starforge.addBlockDecorator",
      async (uri: vscode.Uri, defLine: number) => {
        const document = await vscode.workspace.openTextDocument(uri);
        const edit = new vscode.WorkspaceEdit();
        const indent = document.lineAt(defLine).text.match(/^\s*/)?.[0] ?? "";
        edit.insert(uri, new vscode.Position(defLine, 0), `${indent}@block\n`);
        if (!/^\s*(from\s+starforge\s+import\s+.*\bblock\b|import\s+starforge)/m.test(document.getText())) {
          edit.insert(uri, importInsertPosition(document), "from starforge import block\n");
        }
        await vscode.workspace.applyEdit(edit);
        await document.save(); // save → watcher → palette refresh
      },
    ),

    vscode.languages.registerCodeLensProvider(
      { language: "python", scheme: "file" },
      new AddBlockLensProvider(),
    ),

    // Settings feed the kernel at initialize; restart lazily on change.
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("starforge")) {
        for (const kernel of kernels.values()) kernel.shutdown();
      }
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

/** Where a new `from starforge import block` line belongs: with the existing
 * imports if any, after a module docstring otherwise, else line 0. */
function importInsertPosition(document: vscode.TextDocument): vscode.Position {
  const text = document.getText();
  const firstImport = text.match(/^(?:import|from)\s+\S+/m);
  if (firstImport && firstImport.index !== undefined) {
    return document.positionAt(firstImport.index);
  }
  const docstring = text.match(/^(?:#[^\n]*\n|\s*\n)*("""|''')[\s\S]*?\1\s*\n/);
  if (docstring && docstring.index !== undefined) {
    return document.positionAt(docstring.index + docstring[0].length);
  }
  return new vscode.Position(0, 0);
}

/** "⊕ Add to *Forge palette" above undecorated module-level functions. The
 * command writes the @block decorator into the source — code stays the single
 * source of truth; there is no shadow registry. */
class AddBlockLensProvider implements vscode.CodeLensProvider {
  provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
    const lenses: vscode.CodeLens[] = [];
    for (let line = 0; line < document.lineCount; line++) {
      const text = document.lineAt(line).text;
      if (!/^(?:async\s+)?def\s+[A-Za-z_]\w*\s*\(/.test(text)) continue;
      if (text.startsWith(" ") || text.startsWith("\t")) continue; // module-level only
      // Walk the contiguous decorator stack above; skip if @block is present.
      let cursor = line - 1;
      let decorated = false;
      while (cursor >= 0) {
        const above = document.lineAt(cursor).text.trim();
        if (!above.startsWith("@") && above !== "") break;
        if (/^@(?:\w+\.)?block\b/.test(above)) decorated = true;
        cursor--;
      }
      if (decorated) continue;
      lenses.push(
        new vscode.CodeLens(new vscode.Range(line, 0, line, 0), {
          title: "⊕ Add to *Forge palette",
          command: "starforge.addBlockDecorator",
          arguments: [document.uri, line],
        }),
      );
    }
    return lenses;
  }
}

class CanvasEditorProvider implements vscode.CustomTextEditorProvider {
  private figurePanel: vscode.WebviewPanel | undefined;

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly output: vscode.OutputChannel,
  ) {}

  /** Interactive figures get a dedicated editor-tab webview whose HTML IS
   * the plotly document. Iframes inside the canvas webview are a dead end:
   * VS Code's resource service worker refuses opaque/sandboxed iframe
   * requests, so the document never loads. A panel has no middleman. */
  private showFigurePanel(title: string, html: string): void {
    if (!this.figurePanel) {
      this.figurePanel = vscode.window.createWebviewPanel(
        "starforge.figure",
        title,
        { viewColumn: vscode.ViewColumn.Beside, preserveFocus: false },
        { enableScripts: true, retainContextWhenHidden: true },
      );
      this.figurePanel.onDidDispose(() => (this.figurePanel = undefined));
    }
    // The artifact carries its own inline plotly.js; the CSP (scoped to this
    // panel only) must allow it. Same trust level as the pipeline that
    // produced it — we already execute that user's code.
    const csp =
      `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; ` +
      `script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:;">`;
    this.figurePanel.title = title;
    this.figurePanel.webview.html = html.includes("<head>")
      ? html.replace("<head>", `<head>${csp}`)
      : csp + html;
    this.figurePanel.reveal(vscode.ViewColumn.Beside, false);
  }

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

    // Async continuations (kernel events, post-run refreshes) routinely
    // outlive the panel; posting to a disposed webview throws. Guard at the
    // single choke point.
    let disposed = false;
    const post = (type: string, payload: any) => {
      if (disposed) return;
      try {
        void panel.webview.postMessage({ type, payload });
      } catch {
        // disposed between check and post — ignore
      }
    };

    // Full-document replaces MUST be serialized: two concurrent applyEdits
    // compute ranges against different document versions and splice garbage.
    let writeQueue: Promise<void> = Promise.resolve();
    let pendingWrites = 0;

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
      // While webview commits are in flight, the text is about to change —
      // pushing now would echo a STALE doc over the webview's newer state
      // (the "blocks vanish until reload" failure mode). The final write
      // triggers its own push.
      if (pendingWrites > 0 || disposed) return;
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
                path: `${checkpoint.dir}/outputs/${a.file}`,
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
            pendingWrites++;
            writeQueue = writeQueue.then(async () => {
              try {
                const edit = new vscode.WorkspaceEdit();
                const text = JSON.stringify(message.payload, null, 2) + "\n";
                // Range computed HERE, inside the queue — always against the
                // current document version.
                const fullRange = new vscode.Range(0, 0, document.lineCount, 0);
                edit.replace(document.uri, fullRange, text);
                await vscode.workspace.applyEdit(edit);
              } finally {
                pendingWrites--;
              }
            });
            await writeQueue;
            if (pendingWrites === 0) void pushDoc(); // guaranteed fresh echo
            break;
          }
          case "run": {
            const doc = parseDoc();
            if (!doc) break;
            try {
              // Desktop semantics: Run saves the pipeline first, always.
              if (document.isDirty) await document.save();
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
                  if (entry.artifact?.file) {
                    entry.artifact.uri = artifactUri(manifest.dir, entry.artifact.file);
                    entry.artifact.path = `${manifest.dir}/outputs/${entry.artifact.file}`;
                  }
                }
                manifest.figures = (manifest.figures ?? []).map((f: any) => ({
                  ...f,
                  uri: artifactUri(manifest.dir, f.file),
                  path: `${manifest.dir}/outputs/${f.file}`,
                }));
              }
              post("manifest", manifest);
            } catch {
              post("manifest", null);
            }
            break;
          }
          case "openExternal": {
            if (typeof message.payload === "string") {
              void vscode.env.openExternal(vscode.Uri.joinPath(folder.uri, message.payload));
            }
            break;
          }
          case "openFigurePanel": {
            const relPath = message.payload?.path;
            if (typeof relPath !== "string") break;
            try {
              const bytes = await vscode.workspace.fs.readFile(vscode.Uri.joinPath(folder.uri, relPath));
              const title = message.payload?.title ?? relPath.split("/").pop() ?? "Interactive figure";
              this.showFigurePanel(title, Buffer.from(bytes).toString("utf-8"));
            } catch (err: any) {
              this.output.appendLine(`[figure] read failed: ${err.message}`);
              void vscode.window.showErrorMessage(`*Forge: could not open figure (${relPath}).`);
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

    panel.onDidDispose(() => {
      disposed = true;
      subscriptions.forEach((s) => s.dispose());
    });
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
        content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}'; img-src ${webview.cspSource} data:;">
  <link rel="stylesheet" href="${style}">
</head>
<body>
  <div id="root"></div>
  <script nonce="${nonce}" src="${script}"></script>
</body>
</html>`;
  }
}
