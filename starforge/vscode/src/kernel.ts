import { ChildProcess, spawn } from "child_process";
import * as vscode from "vscode";

/** NDJSON JSON-RPC client for the *Forge kernel (DESIGN.md §9).
 *
 * The kernel is spawned lazily on first use and killed on idle/dispose so an
 * idle workspace costs nothing. All requests funnel through one child
 * process per workspace folder.
 */

export interface BlockInfo {
  block_id: string;
  module: string;
  qualname: string;
  file: string;
  lineno: number;
  label: string;
  category: string;
  params: { name: string; annotation: string | null; default_repr: string | null; has_default: boolean }[];
  outputs: string[];
  returns: string | null;
  doc: string | null;
  source_hash: string;
}

export interface NodeState {
  history_hash: string | null;
  stale: boolean;
  problems: string[];
}

export type Palette = { blocks: BlockInfo[]; errors: Record<string, string[]> };

type Pending = { resolve: (v: any) => void; reject: (e: Error) => void };

const IDLE_SHUTDOWN_MS = 5 * 60 * 1000;

/** Kernel and extension version in lockstep (scripts/version_sync.py). An
 * older installed kernel gets a friendly upgrade nudge instead of mystery
 * protocol failures. */
const MIN_KERNEL_VERSION = "0.1.2";

function versionAtLeast(actual: string, required: string): boolean {
  const a = actual.split(".").map((n) => parseInt(n, 10) || 0);
  const r = required.split(".").map((n) => parseInt(n, 10) || 0);
  for (let i = 0; i < 3; i++) {
    if ((a[i] ?? 0) > (r[i] ?? 0)) return true;
    if ((a[i] ?? 0) < (r[i] ?? 0)) return false;
  }
  return true;
}

export class KernelClient implements vscode.Disposable {
  private proc: ChildProcess | undefined;
  private nextId = 0;
  private pending = new Map<number, Pending>();
  private buffer = "";
  private idleTimer: NodeJS.Timeout | undefined;
  private starting: Promise<void> | undefined;

  readonly onRunEvent = new vscode.EventEmitter<any>();
  readonly onPaletteChanged = new vscode.EventEmitter<Palette>();

  constructor(
    private readonly workspace: vscode.WorkspaceFolder,
    private readonly output: vscode.OutputChannel,
  ) {}

  async resolvePython(): Promise<string> {
    const configured = vscode.workspace.getConfiguration("starforge").get<string>("pythonPath");
    if (configured) return configured;
    try {
      const pythonExt = vscode.extensions.getExtension("ms-python.python");
      if (pythonExt) {
        if (!pythonExt.isActive) await pythonExt.activate();
        const api: any = pythonExt.exports;
        const environment = await api?.environments?.resolveEnvironment(
          api?.environments?.getActiveEnvironmentPath(this.workspace.uri),
        );
        const interpreter = environment?.executable?.uri?.fsPath;
        if (interpreter) return interpreter;
      }
    } catch {
      // fall through to PATH lookup
    }
    return "python";
  }

  private async ensureStarted(): Promise<void> {
    if (this.proc && this.proc.exitCode === null) return;
    if (this.starting) return this.starting;
    this.starting = this.start();
    try {
      await this.starting;
    } finally {
      this.starting = undefined;
    }
  }

  private async start(): Promise<void> {
    const python = await this.resolvePython();
    this.output.appendLine(`[kernel] starting: ${python} -m starforge.kernel`);
    this.proc = spawn(python, ["-m", "starforge.kernel"], {
      cwd: this.workspace.uri.fsPath,
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.proc.stdout!.on("data", (chunk: Buffer) => this.onData(chunk));
    this.proc.stderr!.on("data", (chunk: Buffer) =>
      this.output.appendLine(`[kernel stderr] ${chunk.toString().trimEnd()}`),
    );
    this.proc.on("exit", (code) => {
      this.output.appendLine(`[kernel] exited with code ${code}`);
      for (const { reject } of this.pending.values()) reject(new Error("kernel exited"));
      this.pending.clear();
      this.proc = undefined;
    });
    this.proc.on("error", (err) => {
      this.output.appendLine(`[kernel] spawn failed: ${err.message}`);
      vscode.window.showErrorMessage(
        `*Forge kernel failed to start (${err.message}). Install it with: pip install -e <forge repo>/starforge — or set starforge.pythonPath.`,
      );
    });

    const config = vscode.workspace.getConfiguration("starforge");
    const init = await this.request("initialize", {
      workspace: this.workspace.uri.fsPath,
      settings: {
        pickle_enabled: config.get<boolean>("pickleEnabled") ?? false,
        tier: config.get<string>("stalenessTier") ?? "T2",
        max_checkpoint_mb: config.get<number>("maxCheckpointSizeMB") ?? 2048,
      },
    });
    this.output.appendLine(`[kernel] ready: v${init.kernel_version} (python ${init.python})`);
    if (init.kernel_version && !versionAtLeast(String(init.kernel_version), MIN_KERNEL_VERSION)) {
      void vscode.window.showWarningMessage(
        `*Forge kernel ${init.kernel_version} is older than this extension expects (${MIN_KERNEL_VERSION}+). ` +
          `Some features may misbehave — upgrade with: pip install -U "starforge-kernel[mcp]"`,
      );
    }
  }

  private onData(chunk: Buffer): void {
    this.buffer += chunk.toString("utf-8");
    let newline: number;
    while ((newline = this.buffer.indexOf("\n")) >= 0) {
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      if (!line) continue;
      let frame: any;
      try {
        frame = JSON.parse(line);
      } catch {
        this.output.appendLine(`[kernel] bad frame: ${line.slice(0, 200)}`);
        continue;
      }
      if (frame.id !== undefined && this.pending.has(frame.id)) {
        const { resolve, reject } = this.pending.get(frame.id)!;
        this.pending.delete(frame.id);
        frame.error ? reject(new Error(frame.error.message)) : resolve(frame.result);
      } else if (frame.method === "run/event") {
        this.onRunEvent.fire(frame.params);
      } else if (frame.method === "log") {
        this.output.appendLine(`[run] ${frame.params.text}`);
      }
    }
  }

  private touchIdleTimer(): void {
    if (this.idleTimer) clearTimeout(this.idleTimer);
    this.idleTimer = setTimeout(() => {
      this.output.appendLine("[kernel] idle — shutting down");
      this.shutdown();
    }, IDLE_SHUTDOWN_MS);
  }

  async request(method: string, params: Record<string, any> = {}): Promise<any> {
    if (method !== "initialize") await this.ensureStarted();
    this.touchIdleTimer();
    const id = ++this.nextId;
    const frame = JSON.stringify({ id, method, params }) + "\n";
    return new Promise((resolve, reject) => {
      if (!this.proc?.stdin?.writable) {
        reject(new Error("kernel not running"));
        return;
      }
      this.pending.set(id, { resolve, reject });
      this.proc.stdin.write(frame, "utf-8");
    });
  }

  async scan(): Promise<Palette> {
    const palette = (await this.request("index/scan")) as Palette;
    this.onPaletteChanged.fire(palette);
    return palette;
  }

  async hashes(doc: any): Promise<Record<string, NodeState>> {
    const result = await this.request("pipeline/hashes", { doc });
    return result.nodes;
  }

  async runStart(doc: any, target?: string): Promise<string> {
    const result = await this.request("run/start", target ? { doc, target } : { doc });
    return result.run_id;
  }

  async runCancel(runId: string): Promise<void> {
    await this.request("run/cancel", { run_id: runId });
  }

  async manifest(historyHash: string): Promise<any> {
    return this.request("results/manifest", { history_hash: historyHash });
  }

  async figures(
    historyHashes: string[],
  ): Promise<{ checkpoints: Record<string, { dir: string | null; artifacts: { file: string; kind: string }[] }> }> {
    return this.request("results/figures", { history_hashes: historyHashes });
  }

  async gcCheckpoints(maxMb?: number): Promise<{ freed_bytes: number; deleted: number; remaining_bytes: number }> {
    return this.request("maintenance/gc", maxMb === undefined ? {} : { max_mb: maxMb });
  }

  shutdown(): void {
    if (this.idleTimer) clearTimeout(this.idleTimer);
    const proc = this.proc;
    this.proc = undefined;
    if (proc && proc.exitCode === null) {
      try {
        proc.stdin?.write(JSON.stringify({ id: ++this.nextId, method: "shutdown", params: {} }) + "\n");
      } catch {
        // already gone
      }
      setTimeout(() => proc.kill(), 1500);
    }
  }

  dispose(): void {
    this.shutdown();
    this.onRunEvent.dispose();
    this.onPaletteChanged.dispose();
  }
}
