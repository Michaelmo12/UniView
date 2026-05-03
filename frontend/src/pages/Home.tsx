import { useEffect, useRef, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { useSSEStream } from "../hooks/useTrackingStream";
import type { SSEStatus } from "../hooks/useTrackingStream";
import type { StreamPayload, TrackEntry } from "../types/tracking";
import "./Home.css";

// The 4 drone IDs we expect — matches drone_ids=[3,4,6,7] in algorithm config.
// Each ID maps to one cell in the 2×2 grid.
const DRONE_IDS = ["3", "4", "6", "7"];

interface DroneCellProps {
  id: string;
  payload: StreamPayload | null;
  sseStatus: SSEStatus;
}

function DroneCell({ id, payload, sseStatus }: DroneCellProps) {
  // isStale: payload was received before but no update for >3s → drone went silent
  const lastSeenRef = useRef<number | null>(null);
  const [isStale, setIsStale] = useState(false);

  useEffect(() => {
    if (payload !== null) {
      lastSeenRef.current = Date.now();
      setTimeout(() => setIsStale(false), 0);
    }
  }, [payload]);

  useEffect(() => {
    const interval = setInterval(() => {
      if (lastSeenRef.current !== null && Date.now() - lastSeenRef.current > 3000) {
        setIsStale(true);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const isLive = payload !== null && !isStale;

  // canvasRef — direct handle to the <canvas> DOM element for 2D drawing
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Stale-frame guard: JPEG decoding is async (img.onload fires later).
  // If two payloads arrive quickly, a slow first decode could overwrite the
  // newer frame.  We store the "latest" payload here and check inside onload
  // — if it's no longer the latest, we discard the draw.
  const latestPayloadRef = useRef<StreamPayload | null>(null);

  // FPS counter — measures time between consecutive payloads for this drone
  const lastFrameTimeRef = useRef<number | null>(null);
  const [fps, setFps] = useState<number | null>(null);

  // Actual resolution read from the decoded JPEG (naturalWidth x naturalHeight)
  const [resolution, setResolution] = useState<string | null>(null);

  useEffect(() => {
    // Measure FPS — time between this payload and the previous one
    if (payload) {
      const now = performance.now();
      if (lastFrameTimeRef.current !== null) {
        const delta = now - lastFrameTimeRef.current;
        setFps(Math.round(1000 / delta));
      }
      lastFrameTimeRef.current = now;
    }

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
      setResolution(`${img.naturalWidth}×${img.naturalHeight}`);

      // Scale factors: bounding boxes from the algorithm are in original image
      // coordinates; we need to map them to canvas pixel coordinates
      const scaleX = canvas.width / img.naturalWidth;
      const scaleY = canvas.height / img.naturalHeight;

      // Per-ID color palette — hue spread across the visible spectrum
      const TRACK_COLORS = [
        "#0ea5e9", // sky blue
        "#ec4899", // pink
        "#f97316", // orange
        "#8b5cf6", // violet
        "#10b981", // emerald
        "#eab308", // yellow
        "#ef4444", // red
        "#06b6d4", // cyan
        "#a21caf", // purple
        "#16a34a", // green
      ];
      const colorFor = (id: number) => TRACK_COLORS[id % TRACK_COLORS.length];

      payload.tracks.forEach((track: TrackEntry) => {
        const isSingleView = track.global_id === -1;
        const color = isSingleView ? "#f8fafc" : colorFor(track.global_id);
        const rx = track.x * scaleX;
        const ry = track.y * scaleY;
        const rw = track.width * scaleX;
        const rh = track.height * scaleY;

        if (isSingleView) {
          // Dashed bright box — unmatched single-camera detection
          ctx.lineWidth = 2;
          ctx.strokeStyle = color;
          ctx.setLineDash([6, 4]);
          ctx.strokeRect(rx, ry, rw, rh);
          ctx.setLineDash([]);
        } else {
          ctx.lineWidth = 1;
          // Solid colored box for confirmed tracked person
          ctx.strokeStyle = color;
          ctx.setLineDash([]);
          ctx.strokeRect(rx, ry, rw, rh);

          // Small label pill at bottom-left of bounding box
          const label = `ID:${track.global_id}`;
          ctx.font = "600 14px monospace";
          const textW = ctx.measureText(label).width;
          const padX = 4;
          const padY = 3;
          const labelH = 18;
          const lx = rx;
          const ly = ry + rh + 1;

          ctx.fillStyle = color + "33";
          ctx.fillRect(lx, ly, textW + padX * 2, labelH);
          ctx.strokeStyle = color;
          ctx.lineWidth = 0.5;
          ctx.strokeRect(lx, ly, textW + padX * 2, labelH);
          ctx.fillStyle = color;
          ctx.font = "600 14px monospace";
          ctx.fillText(label, lx + padX, ly + labelH - padY - 1);
        }
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
        {isLive || isStale ? (
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

        {sseStatus === "reconnecting" && (
          <div className="drone-cell__overlay drone-cell__overlay--reconnecting">
            ⟳ Reconnecting...
          </div>
        )}

        {isStale && sseStatus !== "reconnecting" && (
          <div className="drone-cell__overlay drone-cell__overlay--nosignal">
            NO SIGNAL
          </div>
        )}
      </div>

      <div className="drone-cell__footer">
        <span>{resolution ?? "--×--"}</span>
        <span>CH-{id.padStart(2, "0")}</span>
        <span>{fps !== null ? `${fps}FPS` : "--FPS"}</span>
        <span style={{ color: "rgba(250,204,21,0.7)" }}>DET:{payload ? payload.tracks.length : "--"}</span>
        <span style={{ color: "rgba(250,204,21,0.7)" }}>
          IDS:{payload
            ? [...new Set(payload.tracks.filter(t => t.global_id !== -1).map(t => t.global_id))].length
            : "--"}
        </span>
      </div>
    </div>
  );
}

// Home is the live monitoring page.
// It opens the SSE connection and passes each drone's latest payload into the
// matching DroneCell.  Cells whose drone hasn't sent anything yet receive null
// and render the "NO SIGNAL" state.
function Home() {
  const { token, user } = useAuth();
  const [toast, setToast] = useState<string | null>(null);

  // Show welcome toast only once per login session
  useEffect(() => {
    if (user?.full_name && !sessionStorage.getItem("welcome_shown")) {
      sessionStorage.setItem("welcome_shown", "1");
      setToast(`Welcome back, ${user.full_name}`);
      const t = setTimeout(() => setToast(null), 4000);
      return () => clearTimeout(t);
    }
  }, []);

  // frames: Map<droneId, StreamPayload> — one entry per drone, updates on every SSE event
  const { frames, status } = useSSEStream(token);

  return (
    <div className="uniview-dashboard">
      {toast && <div className="uniview-toast">{toast}</div>}
      <main className="uniview-grid-wrapper">
        {/* 2×2 grid — one cell per drone */}
        <div className="uniview-grid">
          {DRONE_IDS.map((id) => (
            <DroneCell
              key={id}
              id={id}
              payload={frames.get(id) ?? null}
              sseStatus={status}
            />
          ))}
        </div>
      </main>
    </div>
  );
}

export default Home;
