/**
 * CameraGrid
 *
 * Renders all 8 drone camera feeds in a responsive 4-column grid.
 * Passes the relevant frame and tracked persons to each CameraView.
 */
import type { TrackingMessage } from "../../types/tracking";
import { CameraView } from "./CameraView";

// All 8 drone IDs (strings, matching the JSON keys from the algorithm)
const DRONE_IDS = ["1", "2", "3", "4", "5", "6", "7", "8"];

interface CameraGridProps {
  message: TrackingMessage | null;
}

export function CameraGrid({ message }: CameraGridProps) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
        gap: 4,
        background: "#000",
      }}
    >
      {DRONE_IDS.map((droneId) => (
        <CameraView
          key={droneId}
          droneId={droneId}
          frameData={message?.frames[droneId] ?? null}
          trackedPersons={message?.tracked_persons ?? []}
        />
      ))}
    </div>
  );
}
