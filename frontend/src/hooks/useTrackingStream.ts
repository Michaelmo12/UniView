/**
 * useSSEStream
 *
 * Connects to the gateway SSE endpoint GET /stream/live?token=<jwt>.
 * Returns a Map<droneId, StreamPayload> that updates as payloads arrive.
 *
 * One payload per drone per frame — the Map accumulates all drones so
 * the Home page can render up to 8 cameras simultaneously.
 *
 * EventSource auto-reconnects on error (browser-native).
 * JWT in query param — EventSource does not support custom headers.
 */
import { useEffect, useRef, useState } from "react";
import type { StreamPayload } from "../types/tracking";

// Gateway base URL — falls back to localhost if env var not set
const SSE_BASE =
  (import.meta.env.VITE_GATEWAY_URL as string | undefined) || "http://localhost:8080";

export type SSEStatus = "connecting" | "connected" | "reconnecting";

export interface SSEStreamResult {
  frames: Map<string, StreamPayload>;
  status: SSEStatus;
}

export function useSSEStream(token: string | null): SSEStreamResult {
  // Map<drone_id, StreamPayload> — holds the latest frame+tracks for each drone.
  const [frames, setFrames] = useState<Map<string, StreamPayload>>(new Map());
  const [status, setStatus] = useState<SSEStatus>("connecting");

  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!token) return;

    const url = `${SSE_BASE}/stream/live?token=${encodeURIComponent(token)}`;

    const es = new EventSource(url);
    esRef.current = es;

    es.onmessage = (event: MessageEvent) => {
      try {
        const payload = JSON.parse(event.data as string) as StreamPayload;
        setStatus("connected");
        setFrames((prev) => new Map(prev).set(payload.drone_id, payload));
      } catch {
        // ignore malformed events
      }
    };

    es.onerror = () => {
      // EventSource will auto-reconnect — show reconnecting state until next message
      setStatus("reconnecting");
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [token]);

  return { frames, status };
}
