/**
 * TypeScript types for SSE stream payloads from the gateway /stream/live endpoint.
 * Mirrors algorithm/src/pipeline/output_formatter.py StreamPayload schema.
 *
 * One StreamPayload arrives per drone per frame.
 * Frontend accumulates them in a Map<droneId, StreamPayload> keyed by drone_id.
 */

export interface TrackEntry {
  global_id: number;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  state: string;
  frames_tracked: number;
}

export interface StageTimings {
  detection: number;
  features: number;
  fusion: number;
  reconstruction: number;
  tracking: number;
  total: number;
}

export interface StreamPayload {
  timestamp: string;           // ISO-8601 UTC
  drone_id: string;            // e.g. "1"
  frame_base64: string;        // raw base64, no data: prefix
  tracks: TrackEntry[];        // CONFIRMED tracks visible on this drone
  active_drones_count: number;
  total_reid_matches: number;
  pipeline_latency_ms: number;
  avg_confidence: number;
  stage_timings_ms: StageTimings;
}
