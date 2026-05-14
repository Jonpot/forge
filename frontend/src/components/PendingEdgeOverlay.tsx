import { createPortal } from "react-dom";

interface PendingEdgeOverlayProps {
  /** Screen-space center of the originating output handle. */
  source: { x: number; y: number };
  /** Screen-space drop point — the live target of the ghost edge. */
  target: { x: number; y: number };
}

/**
 * SVG overlay rendered above the canvas but beneath the block picker.
 * Draws an animated dashed bezier from the source handle to the drop point
 * while the user is choosing a block, so the in-progress edge feels
 * preserved rather than vanishing the moment the picker opens.
 *
 * Coordinates are pure screen-space (the picker captures clicks, so the
 * user can't pan/zoom while it's open).
 */
export function PendingEdgeOverlay({ source, target }: PendingEdgeOverlayProps) {
  const dx = Math.max(40, Math.abs(target.x - source.x) / 2);
  const c1x = source.x + dx;
  const c1y = source.y;
  const c2x = target.x - dx;
  const c2y = target.y;
  const path = `M ${source.x},${source.y} C ${c1x},${c1y} ${c2x},${c2y} ${target.x},${target.y}`;

  return createPortal(
    <svg
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 z-[70]"
      width="100%"
      height="100%"
    >
      {/* Faint shadow layer for depth */}
      <path
        d={path}
        fill="none"
        stroke="rgba(99, 102, 241, 0.25)"
        strokeWidth={6}
        strokeLinecap="round"
      />
      {/* Animated dashed accent */}
      <path
        d={path}
        fill="none"
        stroke="#818cf8"
        strokeWidth={2}
        strokeLinecap="round"
        strokeDasharray="8 6"
        className="animate-dash-flow"
      />
      {/* End dot to mark the drop point */}
      <circle
        cx={target.x}
        cy={target.y}
        r={4}
        fill="#818cf8"
        opacity={0.9}
      />
    </svg>,
    document.body,
  );
}
