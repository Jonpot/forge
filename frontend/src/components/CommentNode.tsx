import {
  memo,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ChangeEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { createPortal } from "react-dom";
import {
  NodeResizer,
  useReactFlow,
  useViewport,
  type NodeProps,
  type ResizeParams,
} from "@xyflow/react";
import type { CommentNodeData } from "@/hooks/usePipeline";
import { useBackdropBlur } from "@/hooks/useBackdropBlur";
import {
  buildCommentSwatchBackground,
  buildCommentTheme,
  COMMENT_COLOR_OPTIONS,
  hexToRgb,
  hsvToHex,
  isCommentPaletteColor,
  normalizeCommentColor,
  rgbToHsv,
} from "@/utils/commentColors";

const MENU_PANEL_WIDTH = 244;
const MENU_PANEL_HEIGHT = 136;
const MENU_ANCHOR_GAP = 8;
const MENU_VIEWPORT_MARGIN = 12;

type MenuPosition = {
  left: number;
  top: number;
};

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function readSaturationValueFromPointer(
  event: PointerEvent | ReactPointerEvent,
  element: HTMLDivElement,
) {
  const rect = element.getBoundingClientRect();
  const saturation = clamp01((event.clientX - rect.left) / rect.width);
  const value = clamp01(1 - (event.clientY - rect.top) / rect.height);
  return { saturation, value };
}

function readHueFromPointer(
  event: PointerEvent | ReactPointerEvent,
  element: HTMLDivElement,
) {
  const rect = element.getBoundingClientRect();
  const ratio = clamp01((event.clientY - rect.top) / rect.height);
  return ratio * 360;
}

export const CommentNode = memo(function CommentNode({
  id,
  data,
  height,
  width,
  selected,
}: NodeProps) {
  const { title, description, color } = data as CommentNodeData;
  const { setNodes } = useReactFlow();
  const { zoom } = useViewport();
  const backdropBlur = useBackdropBlur();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const saturationRef = useRef<HTMLDivElement | null>(null);
  const hueRef = useRef<HTMLDivElement | null>(null);
  const [showColorMenu, setShowColorMenu] = useState(false);
  const [showCustomEditor, setShowCustomEditor] = useState(false);
  const [menuPosition, setMenuPosition] = useState<MenuPosition>({
    left: MENU_VIEWPORT_MARGIN,
    top: MENU_VIEWPORT_MARGIN,
  });
  const theme = buildCommentTheme(color, selected || showColorMenu);
  const normalizedColor = normalizeCommentColor(color);
  const usingCustomColor =
    normalizedColor !== null && !isCommentPaletteColor(normalizedColor);
  const currentHsv = rgbToHsv(hexToRgb(theme.color));
  const [hexDraft, setHexDraft] = useState(theme.color);
  const commentHeight = typeof height === "number" ? height : 160;
  const commentWidth = typeof width === "number" ? width : 160;
  const commentConstraint = Math.min(commentHeight, commentWidth);
  const titleMaxScreenSize = clamp(Math.min(commentConstraint * 0.12, commentWidth * 0.009), 11, 24);
  const titleMinScreenSize = clamp(commentConstraint * 0.055, 6, 10);
  const titleTargetScreenSize = clamp(
    commentConstraint * zoom * 0.12,
    titleMinScreenSize,
    titleMaxScreenSize,
  );
  const titleFontSize = clamp(
    titleTargetScreenSize / Math.max(zoom, 0.1),
    11,
    commentConstraint * 0.14,
  );
  const titleLineHeight = titleFontSize * 1.45;

  useEffect(() => {
    setHexDraft(theme.color.toUpperCase());
  }, [theme.color]);

  useEffect(() => {
    if (!showColorMenu) return;

    const handlePointerDown = (event: PointerEvent) => {
      if (!(event.target instanceof Node)) return;
      const insideContainer = containerRef.current?.contains(event.target) ?? false;
      const insideMenu = menuRef.current?.contains(event.target) ?? false;
      if (!insideContainer && !insideMenu) {
        setShowColorMenu(false);
        setShowCustomEditor(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setShowColorMenu(false);
        setShowCustomEditor(false);
      }
    };

    window.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [showColorMenu]);

  useLayoutEffect(() => {
    if (!showColorMenu) return;

    let frameId = 0;
    const updatePosition = () => {
      const trigger = triggerRef.current;
      if (!trigger) {
        frameId = window.requestAnimationFrame(updatePosition);
        return;
      }

      const rect = trigger.getBoundingClientRect();
      const nextLeft = Math.round(
        Math.max(
          MENU_VIEWPORT_MARGIN,
          Math.min(
            rect.right - MENU_PANEL_WIDTH,
            window.innerWidth - MENU_PANEL_WIDTH - MENU_VIEWPORT_MARGIN,
          ),
        ),
      );

      const preferredBelow = rect.bottom + MENU_ANCHOR_GAP;
      const preferredAbove = rect.top - MENU_PANEL_HEIGHT - MENU_ANCHOR_GAP;
      const nextTop = Math.round(
        Math.max(
          MENU_VIEWPORT_MARGIN,
          Math.min(
            preferredBelow + MENU_PANEL_HEIGHT <=
            window.innerHeight - MENU_VIEWPORT_MARGIN
              ? preferredBelow
              : preferredAbove,
            window.innerHeight - MENU_PANEL_HEIGHT - MENU_VIEWPORT_MARGIN,
          ),
        ),
      );

      setMenuPosition((current) =>
        current.left === nextLeft && current.top === nextTop
          ? current
          : { left: nextLeft, top: nextTop },
      );

      frameId = window.requestAnimationFrame(updatePosition);
    };

    updatePosition();
    return () => window.cancelAnimationFrame(frameId);
  }, [showColorMenu]);

  const stopPropagation = useCallback(
    (event: { stopPropagation(): void }) => {
      event.stopPropagation();
    },
    [],
  );

  const stopPointerDown = useCallback(
    (event: { stopPropagation(): void; preventDefault(): void }) => {
      event.stopPropagation();
      event.preventDefault();
    },
    [],
  );

  const updateField = useCallback(
    (field: "title" | "description" | "color", value: string) => {
      setNodes((nodes) =>
        nodes.map((node) =>
          node.id === id ? { ...node, data: { ...node.data, [field]: value } } : node,
        ),
      );
    },
    [id, setNodes],
  );

  const applyColor = useCallback(
    (nextColor: string) => {
      updateField("color", nextColor);
    },
    [updateField],
  );

  const selectPresetColor = useCallback(
    (nextColor: string) => {
      applyColor(nextColor);
    },
    [applyColor],
  );

  const updateSaturationValue = useCallback(
    (saturation: number, value: number) => {
      applyColor(
        hsvToHex({
          h: currentHsv.h,
          s: saturation,
          v: value,
        }),
      );
    },
    [applyColor, currentHsv.h],
  );

  const updateHue = useCallback(
    (hue: number) => {
      applyColor(
        hsvToHex({
          h: hue,
          s: currentHsv.s,
          v: currentHsv.v,
        }),
      );
    },
    [applyColor, currentHsv.s, currentHsv.v],
  );

  const handleSaturationPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      event.stopPropagation();
      event.preventDefault();
      const element = saturationRef.current;
      if (!element) return;

      const updateFromEvent = (pointerEvent: PointerEvent | ReactPointerEvent) => {
        const { saturation, value } = readSaturationValueFromPointer(pointerEvent, element);
        updateSaturationValue(saturation, value);
      };

      updateFromEvent(event);
      const onMove = (moveEvent: PointerEvent) => updateFromEvent(moveEvent);
      const onUp = () => {
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    },
    [updateSaturationValue],
  );

  const handleHuePointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      event.stopPropagation();
      event.preventDefault();
      const element = hueRef.current;
      if (!element) return;

      const updateFromEvent = (pointerEvent: PointerEvent | ReactPointerEvent) => {
        updateHue(readHueFromPointer(pointerEvent, element));
      };

      updateFromEvent(event);
      const onMove = (moveEvent: PointerEvent) => updateFromEvent(moveEvent);
      const onUp = () => {
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    },
    [updateHue],
  );

  const handleHexDraftChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const next = event.target.value.toUpperCase();
      setHexDraft(next);
      const normalized = normalizeCommentColor(next);
      if (normalized) {
        applyColor(normalized);
      }
    },
    [applyColor],
  );

  const handleHexDraftBlur = useCallback(() => {
    setHexDraft(theme.color.toUpperCase());
  }, [theme.color]);

  const handleResize = useCallback(
    (_: unknown, params: ResizeParams) => {
      setNodes((nodes) =>
        nodes.map((node) =>
          node.id === id
            ? {
                ...node,
                position: { x: params.x, y: params.y },
                style: {
                  ...node.style,
                  width: params.width,
                  height: params.height,
                },
              }
            : node,
        ),
      );
    },
    [id, setNodes],
  );

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <NodeResizer
        isVisible={selected}
        minWidth={150}
        minHeight={80}
        onResize={handleResize}
        color={theme.color}
        handleStyle={{
          width: 9,
          height: 9,
          borderRadius: 2,
          background: theme.color,
          border: "1.5px solid rgba(255,255,255,0.28)",
        }}
        lineStyle={{
          borderColor: theme.resizerLine,
          borderWidth: 1,
          borderStyle: "dashed",
        }}
      />

      <div
        className="w-full h-full rounded-2xl flex flex-col overflow-hidden border-2 transition-[border-color,box-shadow,background] duration-150"
        style={{
          background: theme.background,
          borderColor: theme.border,
          borderStyle: selected ? "solid" : "dashed",
          ...(backdropBlur ? { backdropFilter: "blur(2px)" } : {}),
          boxShadow: theme.shadow,
        }}
      >
        <div className="px-4 pt-3 pb-2 flex items-center gap-3 flex-shrink-0">
          <input
            value={title}
            onChange={(event) => updateField("title", event.target.value)}
            onMouseDown={stopPropagation}
            autoCapitalize="off"
            autoCorrect="off"
            spellCheck={false}
            placeholder="Comment Title"
            className="
              nodrag nopan
              flex-1 min-w-0 bg-transparent
              text-forge-text font-semibold text-sm
              outline-none border-none
              placeholder:text-forge-border
              cursor-text
            "
            style={{
              fontSize: `${titleFontSize}px`,
              lineHeight: `${titleLineHeight}px`,
            }}
          />
          <button
            type="button"
            ref={triggerRef}
            onMouseDown={stopPropagation}
            onPointerDown={stopPointerDown}
            onClick={(event) => {
              event.stopPropagation();
              setShowColorMenu((current) => {
                const next = !current;
                if (!next) {
                  setShowCustomEditor(false);
                }
                return next;
              });
            }}
            aria-label="Select comment shading"
            aria-expanded={showColorMenu}
            className="nodrag nopan h-[18px] w-[18px] flex-shrink-0 rounded-[6px] border transition-[box-shadow,border-color] duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forge-accent focus-visible:ring-offset-1 focus-visible:ring-offset-forge-surface"
            style={{
              backgroundColor: theme.color,
              borderColor: showColorMenu ? "rgba(255,255,255,0.72)" : theme.buttonRing,
              boxShadow: showColorMenu
                ? `0 0 0 1px ${theme.buttonRing}`
                : "inset 0 0 0 1px rgba(255,255,255,0.08)",
            }}
          />
        </div>

        <div
          className="mx-4 h-px flex-shrink-0"
          style={{ backgroundColor: theme.separator }}
        />

        <div className="px-4 pt-2 pb-3 flex-1 min-h-0">
          <textarea
            value={description}
            onChange={(event) => updateField("description", event.target.value)}
            onMouseDown={stopPropagation}
            autoCapitalize="off"
            autoCorrect="off"
            spellCheck={false}
            placeholder="Add a description…"
            className="
              nodrag nopan
              w-full h-full
              bg-transparent
              text-forge-text/70 text-xs
              resize-none outline-none border-none
              placeholder:text-forge-border
              cursor-text leading-relaxed
            "
          />
        </div>
      </div>

      {showColorMenu &&
        createPortal(
        <div
          ref={menuRef}
          className="nodrag nopan fixed z-[80] overflow-hidden rounded-[20px] border shadow-2xl"
          style={{
            left: `${menuPosition.left}px`,
            top: `${menuPosition.top}px`,
            width: `${MENU_PANEL_WIDTH}px`,
            height: `${MENU_PANEL_HEIGHT}px`,
            backgroundColor: "rgba(26,29,39,0.98)",
            borderColor: theme.border,
            ...(backdropBlur ? { backdropFilter: "blur(14px)" } : {}),
          }}
          onMouseDown={stopPropagation}
          onPointerDown={stopPropagation}
          onClick={stopPropagation}
          >
            <div
              className="flex h-full transition-transform duration-200 ease-out"
            style={{
              width: `${MENU_PANEL_WIDTH * 2}px`,
              transform: `translateX(${showCustomEditor ? -MENU_PANEL_WIDTH : 0}px)`,
            }}
            >
              <div
              className="flex h-full flex-shrink-0 flex-col px-3 py-2.5"
              style={{ width: `${MENU_PANEL_WIDTH}px` }}
            >
              <div className="flex h-5 items-center justify-between gap-3">
                <span className="text-[11px] font-medium uppercase tracking-[0.14em] text-forge-muted">
                  Shading
                </span>
                <input
                  value={hexDraft}
                  onChange={handleHexDraftChange}
                  onBlur={handleHexDraftBlur}
                  onMouseDown={stopPropagation}
                  onClick={stopPropagation}
                  autoCapitalize="off"
                  autoCorrect="off"
                  spellCheck={false}
                  maxLength={7}
                  className="nodrag nopan w-[76px] border-none bg-transparent p-0 text-right text-[10px] font-mono uppercase tracking-[0.08em] text-forge-muted outline-none transition-colors duration-150 focus:text-forge-text"
                  aria-label="Preset color hex"
                />
              </div>

              <div className="flex flex-1 items-center">
                <div className="grid w-full grid-cols-5 gap-2">
                  {COMMENT_COLOR_OPTIONS.map((option) => {
                    const isActive = normalizeCommentColor(option.value) === theme.color;
                    return (
                      <button
                        key={option.value}
                        type="button"
                        aria-label={`${option.label} comment shading`}
                        onMouseDown={stopPropagation}
                        onPointerDown={stopPropagation}
                        onClick={(event) => {
                          event.stopPropagation();
                          selectPresetColor(option.value);
                        }}
                        className="nodrag nopan aspect-square rounded-[12px] border-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forge-accent focus-visible:ring-offset-1 focus-visible:ring-offset-forge-surface"
                        style={{
                          background: buildCommentSwatchBackground(option.value),
                          border: isActive
                            ? "3px solid rgba(255,255,255,0.92)"
                            : "3px solid transparent",
                          opacity: isActive ? 1 : 0.98,
                        }}
                      />
                    );
                  })}

                  <button
                    type="button"
                    aria-label="Custom comment shading"
                    onMouseDown={stopPropagation}
                    onPointerDown={stopPropagation}
                    onClick={(event) => {
                      event.stopPropagation();
                      setShowCustomEditor(true);
                    }}
                    className="nodrag nopan aspect-square rounded-[12px] border-[3px] border-dotted bg-transparent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forge-accent focus-visible:ring-offset-1 focus-visible:ring-offset-forge-surface"
                    style={{
                      borderColor:
                        showCustomEditor || usingCustomColor
                          ? "rgba(255,255,255,0.72)"
                          : "rgba(255,255,255,0.42)",
                    }}
                    title="Custom color"
                  />
                </div>
              </div>
            </div>

              <div
              className="flex h-full flex-shrink-0 flex-col px-3 py-2.5"
              style={{ width: `${MENU_PANEL_WIDTH}px` }}
            >
              <div className="mb-2 flex h-5 items-center justify-between gap-3">
                <div className="flex items-center">
                  <button
                    type="button"
                    onMouseDown={stopPropagation}
                    onPointerDown={stopPropagation}
                    onClick={(event) => {
                        event.stopPropagation();
                        setShowCustomEditor(false);
                      }}
                    className="nodrag nopan inline-flex h-5 w-5 items-center justify-center text-forge-muted transition-colors duration-150 hover:text-forge-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-forge-accent focus-visible:ring-offset-1 focus-visible:ring-offset-forge-surface"
                    aria-label="Back to preset shading"
                  >
                    <svg
                      aria-hidden="true"
                      viewBox="0 0 16 16"
                      className="h-3.5 w-3.5"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.8"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M10.5 3.5 6 8l4.5 4.5" />
                    </svg>
                  </button>
                </div>
                <input
                  value={hexDraft}
                  onChange={handleHexDraftChange}
                  onBlur={handleHexDraftBlur}
                  onMouseDown={stopPropagation}
                  onClick={stopPropagation}
                  autoCapitalize="off"
                  autoCorrect="off"
                  spellCheck={false}
                  maxLength={7}
                  className="nodrag nopan w-[76px] border-none bg-transparent p-0 text-right text-[10px] font-mono uppercase tracking-[0.08em] text-forge-muted outline-none transition-colors duration-150 focus:text-forge-text"
                  aria-label="Custom color hex"
                />
              </div>

              <div className="flex flex-1 items-center gap-2">
                  <div
                    ref={saturationRef}
                    onPointerDown={handleSaturationPointerDown}
                    className="relative h-[84px] min-w-0 flex-1 cursor-crosshair overflow-hidden rounded-[14px] border border-white/12"
                    style={{
                      backgroundColor: hsvToHex({
                        h: currentHsv.h,
                        s: 1,
                        v: 1,
                      }),
                    }}
                  >
                    <div
                      className="absolute inset-0"
                      style={{
                        background:
                          "linear-gradient(to right, rgba(255,255,255,1), rgba(255,255,255,0))",
                      }}
                    />
                    <div
                      className="absolute inset-0"
                      style={{
                        background:
                          "linear-gradient(to top, rgba(0,0,0,1), rgba(0,0,0,0))",
                      }}
                    />
                    <span
                      className="pointer-events-none absolute h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow-[0_0_0_1px_rgba(0,0,0,0.35)]"
                      style={{
                        left: `${currentHsv.s * 100}%`,
                        top: `${(1 - currentHsv.v) * 100}%`,
                        backgroundColor: theme.color,
                      }}
                    />
                  </div>

                  <div
                    ref={hueRef}
                    onPointerDown={handleHuePointerDown}
                    className="relative h-[84px] w-[14px] flex-shrink-0 cursor-ns-resize rounded-full border border-white/12"
                    style={{
                      background:
                        "linear-gradient(to bottom, #ef4444 0%, #f59e0b 17%, #22c55e 33%, #06b6d4 50%, #3b82f6 67%, #8b5cf6 83%, #ef4444 100%)",
                    }}
                  >
                    <span
                      className="pointer-events-none absolute left-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow-[0_0_0_1px_rgba(0,0,0,0.35)]"
                      style={{
                        top: `${(currentHsv.h / 360) * 100}%`,
                        backgroundColor: hsvToHex({
                          h: currentHsv.h,
                          s: 1,
                          v: 1,
                        }),
                      }}
                    />
                  </div>
              </div>
            </div>
          </div>
        </div>
      ,
      document.body,
    )}
    </div>
  );
});
