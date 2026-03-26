import { useEffect, useRef } from "react";
import { useAuth } from "../context/AuthContext";
import { useSSEStream } from "../hooks/useTrackingStream";
import type { StreamPayload, TrackEntry } from "../types/tracking";
import "./Home.css";

// The 4 drone IDs we expect — matches num_drones=4 in algorithm config.
// Each ID maps to one cell in the 2×2 grid.
const DRONE_IDS = ["1", "2", "3", "4"];

interface DroneCellProps {
  id: string;
  payload: StreamPayload | null; // null = drone not yet seen in SSE stream (offline)
}

// DroneCell renders one camera tile.
// When a payload arrives for this drone it:
//   1. Decodes the base64 JPEG and draws it on a <canvas>
//   2. Overlays bounding boxes + global track IDs for every confirmed track
// When no payload is available it shows a "NO SIGNAL" crosshair instead.
function DroneCell({ id, payload }: DroneCellProps) {
  // isLive drives the CSS variant (corner bracket color, footer color, REC badge vs NO SIGNAL)
  const isLive = payload !== null;

  // canvasRef — direct handle to the <canvas> DOM element for 2D drawing
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Stale-frame guard: JPEG decoding is async (img.onload fires later).
  // If two payloads arrive quickly, a slow first decode could overwrite the
  // newer frame.  We store the "latest" payload here and check inside onload
  // — if it's no longer the latest, we discard the draw.
  const latestPayloadRef = useRef<StreamPayload | null>(null);

  useEffect(() => {
    // Always update the ref first — this is what the stale-frame guard reads
    latestPayloadRef.current = payload;
    if (!payload || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Capture the payload value at the time this effect runs.
    // The closure inside img.onload will compare against latestPayloadRef
    // to know if this payload is still the most recent one.
    const capturedPayload = payload;

    // Build a browser Image from the base64 JPEG — decoding happens async
    const img = new Image();
    img.onload = () => {
      // Stale-frame guard: if a newer payload already arrived while this JPEG
      // was decoding, skip the draw so we don't paint an older frame on top
      if (latestPayloadRef.current !== capturedPayload) return;

      // Fill the canvas with the decoded frame (stretched to canvas size)
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      // Scale factors: bounding boxes from the algorithm are in original image
      // coordinates; we need to map them to canvas pixel coordinates
      const scaleX = canvas.width / img.naturalWidth;
      const scaleY = canvas.height / img.naturalHeight;

      // Draw one green bounding box + "ID N" label per confirmed track
      ctx.strokeStyle = "#00ff88";
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.font = "bold 11px monospace";
      ctx.fillStyle = "#00ff88";

      payload.tracks.forEach((track: TrackEntry) => {
        // Scale box coords from algorithm image space → canvas pixel space
        const rx = track.x * scaleX;
        const ry = track.y * scaleY;
        const rw = track.width * scaleX;
        const rh = track.height * scaleY;
        ctx.strokeRect(rx, ry, rw, rh);
        // Label sits just above the top-left corner of the box
        ctx.fillText(`ID ${track.global_id}`, rx + 2, ry - 4);
      });
    };
    // Trigger decode — setting src starts the async load
    img.src = `data:image/jpeg;base64,${payload.frame_base64}`;
  }, [payload]); // Re-runs every time this drone's payload updates

  return (
    <div className={`drone-cell ${isLive ? "drone-cell--live" : "drone-cell--offline"}`}>
      <div className="drone-cell__scanlines" />
      <div className="drone-cell__corners" />

      <div className="drone-cell__header">
        <span className="drone-cell__id">DRONE-{id.padStart(2, "0")}</span>
        <span className={`drone-cell__status ${isLive ? "drone-cell__status--live" : ""}`}>
          {isLive ? (
            <><span className="drone-cell__rec-dot" />REC</>
          ) : (
            "NO SIGNAL"
          )}
        </span>
      </div>

      <div className="drone-cell__body">
        {isLive ? (
          <canvas
            ref={canvasRef}
            width={640}
            height={360}
            style={{ display: "block", width: "100%", height: "100%", objectFit: "cover" }}
          />
        ) : (
          <div className="drone-cell__nosignal">
            <div className="drone-cell__crosshair" />
          </div>
        )}
      </div>

      <div className="drone-cell__footer">
        <span>1920×1080</span>
        <span>CH-{id.padStart(2, "0")}</span>
        <span>30FPS</span>
      </div>
    </div>
  );
}

// Home is the live monitoring page.
// It opens the SSE connection and passes each drone's latest payload into the
// matching DroneCell.  Cells whose drone hasn't sent anything yet receive null
// and render the "NO SIGNAL" state.
function Home() {
  const { token } = useAuth();

  // frames: Map<droneId, StreamPayload> — one entry per drone, updates on every SSE event
  const frames = useSSEStream(token);

  return (
    <div className="uniview-dashboard">
      <main className="uniview-grid-wrapper">
        {/* 2×2 grid — one cell per drone */}
        <div className="uniview-grid">
          {DRONE_IDS.map((id) => (
            <DroneCell
              key={id}
              id={id}
              // Look up this drone's latest payload; null if not yet received
              payload={frames.get(id) ?? null}
            />
          ))}
        </div>
      </main>
    </div>
  );
}

export default Home;
