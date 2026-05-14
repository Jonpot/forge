import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { BlockSpec } from "@/types/pipeline";
import {
  categoryIcon,
  categoryTextClass,
  useCategoryStyleVersion,
} from "@/utils/categoryStyles";

interface BlockPickerMenuProps {
  blocks: BlockSpec[];
  position: { x: number; y: number };
  onSelect: (spec: BlockSpec) => void;
  onClose: () => void;
}

const PANEL_WIDTH = 300;
const PANEL_HEIGHT = 320;
const PANEL_MARGIN = 8;
const TILE_MIN_HEIGHT = 64;

// Mirrors BlockPalette's category order so the picker presents the same
// hierarchy users already learned from the sidebar.
const CATEGORY_ORDER = [
  "IO",
  "Operator",
  "Combine",
  "Transform",
  "Statistics",
  "Clustering",
  "Factorization",
  "Dimensionality",
  "Visualization",
  "Special",
  "Custom",
];

function groupAndSort(blocks: BlockSpec[]): [string, BlockSpec[]][] {
  const byCategory = new Map<string, BlockSpec[]>();
  for (const spec of blocks) {
    const list = byCategory.get(spec.category);
    if (list) list.push(spec);
    else byCategory.set(spec.category, [spec]);
  }
  for (const list of byCategory.values()) {
    list.sort((a, b) =>
      a.name.localeCompare(b.name, undefined, { sensitivity: "base" }),
    );
  }
  const ordered: [string, BlockSpec[]][] = [];
  for (const category of CATEGORY_ORDER) {
    const list = byCategory.get(category);
    if (list) ordered.push([category, list]);
  }
  // Append any plugin-defined categories not in the known order, alphabetically.
  const extras = [...byCategory.keys()]
    .filter((category) => !CATEGORY_ORDER.includes(category))
    .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }));
  for (const category of extras) {
    ordered.push([category, byCategory.get(category)!]);
  }
  return ordered;
}

export function BlockPickerMenu({
  blocks,
  position,
  onSelect,
  onClose,
}: BlockPickerMenuProps) {
  useCategoryStyleVersion();
  const [search, setSearch] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    };
    const onMouseDown = (e: MouseEvent) => {
      const container = containerRef.current;
      if (container && !container.contains(e.target as Node)) {
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    // Capture-phase so we close before React Flow handles the click
    window.addEventListener("mousedown", onMouseDown, true);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("mousedown", onMouseDown, true);
    };
  }, [onClose]);

  const grouped = useMemo(() => {
    const q = search.trim().toLowerCase();
    const matched = q
      ? blocks.filter((b) => {
          if (b.name.toLowerCase().includes(q)) return true;
          if (b.category.toLowerCase().includes(q)) return true;
          if (b.description && b.description.toLowerCase().includes(q))
            return true;
          return false;
        })
      : blocks;
    return groupAndSort(matched);
  }, [blocks, search]);

  const totalMatches = useMemo(
    () => grouped.reduce((sum, [, specs]) => sum + specs.length, 0),
    [grouped],
  );

  const left = Math.max(
    PANEL_MARGIN,
    Math.min(
      position.x - PANEL_WIDTH / 2,
      window.innerWidth - PANEL_WIDTH - PANEL_MARGIN,
    ),
  );
  const top = Math.max(
    PANEL_MARGIN,
    Math.min(
      position.y - 40,
      window.innerHeight - PANEL_HEIGHT - PANEL_MARGIN,
    ),
  );

  return createPortal(
    <div
      ref={containerRef}
      role="dialog"
      aria-label="Add block"
      className="fixed z-[80] flex flex-col rounded-lg border border-forge-border bg-forge-surface shadow-2xl shadow-black/50 animate-fade-in-scale"
      style={{ left, top, width: PANEL_WIDTH, height: PANEL_HEIGHT }}
      onMouseDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-forge-border flex-shrink-0">
        <span className="text-forge-text text-xs font-semibold uppercase tracking-wider">
          Add Block
        </span>
        <button
          onClick={onClose}
          aria-label="Close"
          className="text-forge-muted hover:text-forge-text transition-colors text-[12px] leading-none px-1"
        >
          ✕
        </button>
      </div>

      {/* Search */}
      <div className="px-2 py-2 border-b border-forge-border flex-shrink-0">
        <div className="relative">
          <span
            className="absolute left-2 top-1/2 -translate-y-1/2 text-forge-muted text-[11px] pointer-events-none"
            aria-hidden="true"
          >
            ⌕
          </span>
          <input
            ref={inputRef}
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search blocks…"
            autoCapitalize="off"
            autoCorrect="off"
            spellCheck={false}
            aria-label="Search blocks"
            className="w-full pl-5 pr-2 py-1 rounded-md bg-forge-bg border border-forge-border text-forge-text text-xs placeholder:text-forge-muted focus:outline-none focus:border-forge-accent transition-colors duration-100"
          />
        </div>
      </div>

      {/* Grouped grid — categories ordered the same way as the palette */}
      <div className="flex-1 overflow-y-auto py-2">
        {totalMatches === 0 ? (
          <p className="text-forge-muted text-xs px-3 py-6 text-center">
            {blocks.length === 0
              ? "No blocks with inputs are available."
              : `No blocks match "${search}"`}
          </p>
        ) : (
          <div className="space-y-3">
            {grouped.map(([category, specs]) => (
              <div key={category}>
                <div
                  className={`flex items-center gap-1.5 px-3 pb-1.5 mb-1.5 border-b border-forge-border/50 text-[11px] font-semibold uppercase tracking-wider ${categoryTextClass(category)}`}
                >
                  <span aria-hidden="true">{categoryIcon(category)}</span>
                  <span className="flex-1 text-left truncate">{category}</span>
                  <span className="text-[10px] font-normal tracking-normal text-forge-muted">
                    {specs.length}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 px-2">
                  {specs.map((spec) => (
                    <button
                      key={spec.key}
                      onClick={() => onSelect(spec)}
                      title={spec.description || spec.name}
                      className="
                        flex flex-col items-start justify-center gap-0.5
                        px-2 py-2 rounded-md
                        bg-forge-bg border border-forge-border
                        text-forge-text text-xs text-left
                        hover:border-forge-accent hover:bg-forge-accent/10
                        hover:shadow-sm hover:shadow-forge-accent/10
                        transition-[colors,box-shadow] duration-150
                      "
                      style={{ minHeight: TILE_MIN_HEIGHT }}
                    >
                      <span className="font-medium truncate w-full">
                        {spec.name}
                      </span>
                      <div className="flex items-center gap-1 text-[10px] text-forge-muted w-full min-w-0">
                        {spec.n_inputs > 1 && (
                          <span className="truncate">
                            {spec.n_inputs} inputs
                          </span>
                        )}
                        {spec.is_custom && (
                          <span
                            className="ml-auto text-purple-400"
                            title="User-installed block"
                          >
                            ★
                          </span>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
}
