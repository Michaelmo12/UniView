/**
 * CameraView
 *
 * Renders one drone camera feed on an HTML canvas.
 * - Draws the raw JPEG frame (base64 data URL) from the algorithm
 * - Overlays bounding boxes with global person IDs (client-side rendering)
 * - NO bboxes are drawn server-side (OUT-02 compliance)
 *
 * Race condition guard: tracks latest frameData string so old img.onload
 * callbacks don't overwrite newer frames.
 */
import { useEffect, useRef } from "react";
import type { TrackedPersonMsg } from "../../types/tracking";

interface Detection {
  bbox: [number, number, number, number];
  globalId: number;
}

interface CameraViewProps {
  droneId: string;
  frameData: string | null; // base64 JPEG data URL or null
  trackedPersons: TrackedPersonMsg[]; // full list — filter to this drone inside
}

export function CameraView({
  droneId,
  frameData,
  trackedPersons,
}: CameraViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const latestFrameRef = useRef<string | null>(null);

  // Build detections for this specific drone
  const detections: Detection[] = trackedPersons
    .filter((p) => droneId in p.detections)
    .map((p) => ({
      bbox: p.detections[droneId].bbox,
      globalId: p.global_id,
    }));

  useEffect(() => {
    if (!canvasRef.current || !frameData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Store latest frame to guard against stale onload callbacks
    latestFrameRef.current = frameData;
    const capturedFrame = frameData;

    const img = new Image();
    img.onload = () => {
      // Skip if a newer frame has already arrived
      if (latestFrameRef.current !== capturedFrame) return;

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      const scaleX = canvas.width / img.naturalWidth;
      const scaleY = canvas.height / img.naturalHeight;

      detections.forEach(({ bbox, globalId }) => {
        const [x1, y1, x2, y2] = bbox;
        const rx = x1 * scaleX;
        const ry = y1 * scaleY;
        const rw = (x2 - x1) * scaleX;
        const rh = (y2 - y1) * scaleY;

        ctx.strokeStyle = "#00ff88";
        ctx.lineWidth = 2;
        ctx.strokeRect(rx, ry, rw, rh);

        ctx.fillStyle = "#00ff88";
        ctx.font = "bold 12px monospace";
        ctx.fillText(`ID ${globalId}`, rx + 2, ry - 4);
      });
    };
    img.src = frameData;
  }, [frameData, detections]);

  return (
    <div style={{ position: "relative", background: "#111" }}>
      <div
        style={{
          position: "absolute",
          top: 4,
          left: 4,
          color: "#00ff88",
          fontSize: 11,
          fontFamily: "monospace",
          background: "rgba(0,0,0,0.5)",
          padding: "1px 4px",
          zIndex: 1,
        }}
      >
        Drone {droneId}
      </div>
      <canvas
        ref={canvasRef}
        width={320}
        height={180}
        style={{ display: "block", width: "100%", height: "auto" }}
      />
    </div>
  );
}
