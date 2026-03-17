/**
 * TypeScript interfaces for WebSocket tracking messages from the algorithm.
 * Mirrors the JSON schema produced by the algorithm pipeline output.
 */

export interface DetectionInfo {
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] pixels
}

export interface TrackedPersonMsg {
  global_id: number;
  position: [number, number, number]; // [x, y, z] in meters (world coordinates)
  state: "CONFIRMED";
  frames_tracked: number;
  detections: Record<string, DetectionInfo>; // droneId (string) -> detection
}

export interface TrackingMessage {
  frame_num: number;
  timestamp: number;
  frames: Record<string, string>; // droneId (string) -> base64 JPEG data URL
  tracked_persons: TrackedPersonMsg[];
}
