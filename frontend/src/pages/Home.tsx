import { useEffect, useRef, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { useSSEStream } from "../hooks/useTrackingStream";
import type { SSEStatus } from "../hooks/useTrackingStream";
import type { StreamPayload, TrackEntry } from "../types/tracking";
import "./Home.css";

// The 4 drone IDs we expect — matches drone_ids=[3,4,6,7] in algorithm config.
const DRONE_IDS = ["3", "4", "6", "7"];

// Per-ID color palette — hue spread across the visible spectrum
const TRACK_COLORS = [
  "#0ea5e9",
  "#ec4899",
  "#f97316",
  "#8b5cf6",
  "#10b981",
  "#eab308",
  "#ef4444",
  "#06b6d4",
  "#a21caf",
  "#16a34a",
];
const colorFor = (id: number) => TRACK_COLORS[id % TRACK_COLORS.length];

// One entry in the hit-box list — stored after every bbox redraw so click
// handler can do point-in-rect tests without touching canvas pixels
interface HitBox {
  rx: number;
  ry: number;
  rw: number;
  rh: number;
  track: TrackEntry;
}

// What the click-popup shows for a selected track
interface SelectedTrack {
  track: TrackEntry;
  // Canvas-space position of the box top-left, used to place the popup
  rx: number;
  ry: number;
  rw: number;
  rh: number;
}

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

  // Two separate canvas refs — video layer and bbox overlay layer
  const videoRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  // Stale-frame guard: JPEG decoding is async. If two payloads arrive quickly,
  // a slow first decode could overwrite the newer frame. We compare against
  // latestPayloadRef inside onload to detect and discard stale decodes.
  const latestPayloadRef = useRef<StreamPayload | null>(null);

  // Scale factors computed by the frame effect, read by the bbox effect.
  // null until the first JPEG has decoded — bbox effect skips drawing if null.
  const scaleRef = useRef<{ x: number; y: number } | null>(null);

  // Hit-box list rebuilt after every bbox redraw — read by the click handler
  const hitBoxesRef = useRef<HitBox[]>([]);

  // FPS counter
  const lastFrameTimeRef = useRef<number | null>(null);
  const [fps, setFps] = useState<number | null>(null);

  // Resolution read from decoded JPEG
  const [resolution, setResolution] = useState<string | null>(null);

  // The track the user clicked — drives the popup
  const [selected, setSelected] = useState<SelectedTrack | null>(null);

  // ── Effect 1: decode JPEG → draw to video canvas ──────────────────────────
  useEffect(() => {
    if (payload) {
      const now = performance.now();
      if (lastFrameTimeRef.current !== null) {
        const delta = now - lastFrameTimeRef.current;
        setFps(Math.round(1000 / delta));
      }
      lastFrameTimeRef.current = now;
    }

    latestPayloadRef.current = payload;
    if (!payload || !videoRef.current) return;

    const canvas = videoRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const capturedPayload = payload;

    const img = new Image();
    img.onload = () => {
      // Stale-frame guard: discard if a newer payload already arrived
      if (latestPayloadRef.current !== capturedPayload) return;
      if (!img.naturalWidth || !img.naturalHeight) return;

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      setResolution(`${img.naturalWidth}×${img.naturalHeight}`);

      // Store scale so the bbox effect can use it without waiting for onload
      scaleRef.current = {
        x: canvas.width / img.naturalWidth,
        y: canvas.height / img.naturalHeight,
      };
    };
    img.src = `data:image/jpeg;base64,${payload.frame_base64}`;
  }, [payload]);

  // ── Effect 2: draw bboxes onto the overlay canvas ─────────────────────────
  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Always clear the overlay first — even when payload is null
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    hitBoxesRef.current = [];

    if (!payload || !scaleRef.current) return;

    const { x: scaleX, y: scaleY } = scaleRef.current;
    const newHitBoxes: HitBox[] = [];

    payload.tracks.forEach((track: TrackEntry) => {
      const isSingleView = track.global_id === -1;
      const color = isSingleView ? "#f8fafc" : colorFor(track.global_id);

      const rx = track.x * scaleX;
      const ry = track.y * scaleY;
      const rw = track.width * scaleX;
      const rh = track.height * scaleY;

      // Skip coasting tracks — backend emits x=y=w=h=0 when no detection matched
      if (rw <= 0 || rh <= 0) return;

      // Glow effect — makes strokes visible on any background color
      ctx.shadowBlur = 6;
      ctx.shadowColor = color;

      if (isSingleView) {
        // Dashed bright box — unmatched single-camera detection
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = color;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(rx, ry, rw, rh);
        ctx.setLineDash([]);
      } else {
        // Solid colored box for confirmed tracked person
        ctx.lineWidth = 2;
        ctx.strokeStyle = color;
        ctx.setLineDash([]);
        ctx.strokeRect(rx, ry, rw, rh);

        // Label pill — above box, falls back to inside-top if near top edge
        const label = `ID:${track.global_id}`;
        ctx.font = "600 13px monospace";
        ctx.shadowBlur = 0;
        const textW = ctx.measureText(label).width;
        const padX = 5;
        const padY = 3;
        const labelH = 18;
        const lx = rx;
        const ly = ry >= labelH + 2 ? ry - labelH - 1 : ry + 1;

        // Solid background so label is readable when boxes overlap
        ctx.fillStyle = color;
        ctx.fillRect(lx, ly, textW + padX * 2, labelH);
        ctx.fillStyle = "#0f0f0f";
        ctx.fillText(label, lx + padX, ly + labelH - padY - 1);

        newHitBoxes.push({ rx, ry, rw, rh, track });
      }
    });

    // Reset shadow so it doesn't bleed into next frame
    ctx.shadowBlur = 0;
    hitBoxesRef.current = newHitBoxes;
  }, [payload]);

  // ── Click handler: hit-test against stored hitBoxesRef ────────────────────
  function handleOverlayClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = overlayRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();

    // getBoundingClientRect gives CSS pixels; canvas internal resolution is
    // 640×360, so we scale the click back to canvas coordinates
    const cssToCanvasX = canvas.width / rect.width;
    const cssToCanvasY = canvas.height / rect.height;
    const cx = (e.clientX - rect.left) * cssToCanvasX;
    const cy = (e.clientY - rect.top) * cssToCanvasY;

    // Check each stored hit-box — last match wins (topmost drawn)
    let hit: HitBox | null = null;
    for (const box of hitBoxesRef.current) {
      if (cx >= box.rx && cx <= box.rx + box.rw && cy >= box.ry && cy <= box.ry + box.rh) {
        hit = box;
      }
    }

    if (hit) {
      setSelected({ track: hit.track, rx: hit.rx, ry: hit.ry, rw: hit.rw, rh: hit.rh });
    } else {
      setSelected(null);
    }
  }

  return (
    <div className={`drone-cell ${isLive ? "drone-cell--live" : "drone-cell--offline"}`}>
      <div className="drone-cell__scanlines" />
      <div className="drone-cell__corners" />

      <div className="drone-cell__header">
        <span className="drone-cell__id">DRONE-{id.padStart(2, "0")}</span>
        <div className="drone-cell__header-right">
          <span className="drone-cell__counter">
            DET <strong>{payload ? payload.tracks.length : "--"}</strong>
          </span>
          <span className="drone-cell__counter drone-cell__counter--ids">
            IDS <strong>
              {payload
                ? [...new Set(payload.tracks.filter(t => t.global_id !== -1).map(t => t.global_id))].length
                : "--"}
            </strong>
          </span>
          <span className={`drone-cell__status ${isLive ? "drone-cell__status--live" : ""}`}>
            {isLive ? (
              <><span className="drone-cell__rec-dot" />REC</>
            ) : (
              "NO SIGNAL"
            )}
          </span>
        </div>
      </div>

      <div className="drone-cell__body">
        {isLive || isStale ? (
          // position:relative on this wrapper lets the overlay sit exactly on top
          <div className="drone-cell__canvas-stack">
            {/* Layer 1 — JPEG frames */}
            <canvas
              ref={videoRef}
              width={640}
              height={360}
              className="drone-cell__video-canvas"
            />
            {/* Layer 2 — bbox overlay; pointer-events enabled for click detection */}
            <canvas
              ref={overlayRef}
              width={640}
              height={360}
              className="drone-cell__overlay-canvas"
              onClick={handleOverlayClick}
            />
            {/* Track info popup — shown when user clicks a bbox */}
            {selected && (
              <div
                className="drone-cell__track-popup"
                style={{
                  // Popup width is ~170px; canvas CSS width ≈ container width.
                  // If the box right edge is past 70% of the canvas, anchor to
                  // the left edge of the box instead to avoid right-side clipping.
                  ...((selected.rx + selected.rw) / 640 > 0.70
                    ? { right: `${((640 - selected.rx) / 640) * 100}%` }
                    : { left: `${((selected.rx + selected.rw) / 640) * 100}%` }),
                  top: `${(selected.ry / 360) * 100}%`,
                }}
              >
                <div className="drone-cell__track-popup__row">
                  <span>GLOBAL ID</span><span>{selected.track.global_id}</span>
                </div>
                <div className="drone-cell__track-popup__row">
                  <span>CONFIDENCE</span><span>{(selected.track.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="drone-cell__track-popup__row">
                  <span>STATE</span><span>{selected.track.state}</span>
                </div>
                <div className="drone-cell__track-popup__row">
                  <span>FRAMES</span><span>{selected.track.frames_tracked}</span>
                </div>
                <button className="drone-cell__track-popup__close" onClick={() => setSelected(null)}>✕</button>
              </div>
            )}
          </div>
        ) : (
          <div className="drone-cell__nosignal">
            <div className="drone-cell__crosshair" />
            <span className="drone-cell__nosignal-label">No Signal</span>
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
      </div>
    </div>
  );
}

// Home is the live monitoring page.
// It opens the SSE connection and passes each drone's latest payload into the
// matching DroneCell. Cells whose drone hasn't sent anything yet receive null
// and render the "NO SIGNAL" state.
function Home() {
  const { user, isAuthenticated } = useAuth();
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
  const { frames, status } = useSSEStream(isAuthenticated);

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