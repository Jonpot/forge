import { useEffect, useRef } from "react";
import { checkpointImageUrl } from "@/api/client";

export function isInteractiveArtifact(filename: string): boolean {
  return filename.toLowerCase().endsWith(".html");
}

interface ImagePreviewProps {
  checkpointId: string;
  filename: string;
  onClose: () => void;
}

export function ImageLightbox({ checkpointId, filename, onClose }: ImagePreviewProps) {
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const url = checkpointImageUrl(checkpointId, filename);
  const interactive = isInteractiveArtifact(filename);

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === overlayRef.current) onClose();
      }}
    >
      <div className="relative max-w-[90vw] max-h-[90vh] bg-forge-surface rounded-lg border border-forge-border shadow-2xl overflow-hidden flex flex-col">
        <div className="flex items-center justify-between px-4 py-2 border-b border-forge-border">
          <span className="text-forge-text text-sm font-medium truncate max-w-xs">
            {filename}
          </span>
          <button
            onClick={onClose}
            className="text-forge-muted hover:text-forge-text transition-colors ml-4"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        {interactive ? (
          <iframe
            src={url}
            title={filename}
            className="bg-white rounded-b"
            style={{ width: "85vw", height: "calc(90vh - 52px)", border: 0 }}
          />
        ) : (
          <div className="p-4 overflow-auto max-h-[calc(90vh-52px)]">
            <img
              src={url}
              alt={filename}
              className="max-w-full max-h-full object-contain rounded"
            />
          </div>
        )}
      </div>
    </div>
  );
}

interface ImageThumbnailProps {
  checkpointId: string;
  filename: string;
  onClick: (filename: string) => void;
}

export function ImageThumbnail({ checkpointId, filename, onClick }: ImageThumbnailProps) {
  const url = checkpointImageUrl(checkpointId, filename);
  const interactive = isInteractiveArtifact(filename);
  return (
    <button
      onClick={() => onClick(filename)}
      className="block w-full rounded border border-forge-border overflow-hidden hover:border-forge-accent transition-colors"
      title="Click to expand"
    >
      {interactive ? (
        // pointer-events-none keeps the iframe from swallowing clicks meant
        // for the surrounding "expand" button; the lightbox is where actual
        // interactivity happens.
        <iframe
          src={url}
          title={filename}
          className="w-full bg-white pointer-events-none"
          style={{ height: 160, border: 0 }}
        />
      ) : (
        <img src={url} alt={filename} className="w-full object-cover max-h-40" />
      )}
    </button>
  );
}
