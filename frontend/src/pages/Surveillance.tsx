/**
 * Surveillance Page
 *
 * Main surveillance view showing 8 live camera feeds in a 4x2 grid.
 * Each camera feed displays:
 * - Raw JPEG frame from algorithm (no server-side bbox rendering)
 * - Client-side bbox overlays with global person IDs
 *
 * Connects to gateway WebSocket at /ws/stream using authenticated JWT.
 * Route: /surveillance (protected, requires login)
 */
import { useAuth } from "../context/AuthContext";
import { useTrackingStream } from "../hooks/useTrackingStream";
import { CameraGrid } from "../features/surveillance/CameraGrid";

const GATEWAY_WS_URL =
  (import.meta.env.VITE_GATEWAY_WS_URL as string | undefined) ||
  "ws://localhost:8080/ws/stream";

export default function Surveillance() {
  const { token } = useAuth();
  const message = useTrackingStream(GATEWAY_WS_URL, token);

  const confirmedPersons = message?.tracked_persons ?? [];
  const frameNum = message?.frame_num ?? null;

  return (
    <div
      style={{
        padding: "12px",
        background: "#050505",
        minHeight: "100vh",
        color: "#ccc",
      }}
    >
      <div
        style={{
          marginBottom: 8,
          display: "flex",
          alignItems: "center",
          gap: 16,
        }}
      >
        <h2
          style={{
            margin: 0,
            fontSize: 16,
            fontFamily: "monospace",
            color: "#00ff88",
          }}
        >
          UniView Surveillance
        </h2>
        <span
          style={{ fontSize: 12, fontFamily: "monospace", color: "#666" }}
        >
          {frameNum !== null ? `Frame ${frameNum}` : "Waiting for stream..."}
        </span>
        <span
          style={{ fontSize: 12, fontFamily: "monospace", color: "#666" }}
        >
          {confirmedPersons.length} confirmed person
          {confirmedPersons.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* 8 camera feeds in 4x2 grid */}
      <CameraGrid message={message} />
    </div>
  );
}
