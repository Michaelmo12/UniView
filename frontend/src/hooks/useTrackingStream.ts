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

export function useSSEStream(token: string | null): Map<string, StreamPayload> {
  // Map<drone_id, StreamPayload> — holds the latest frame+tracks for each drone.
  // Starts empty (no drones connected), fills up as payloads arrive from the gateway.
  const [frames, setFrames] = useState<Map<string, StreamPayload>>(new Map());

  // Holds the EventSource object so the cleanup function can close it on unmount.
  // useRef doesn't cause re-renders when it changes — just a stable box to store a value.
  // eventsource is A built-in browser API — like WebSocket but simpler, one direction only (server → browser). You just give it a URL and it connects and listens.
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    // Don't open a connection until the user is logged in and has a JWT
    if (!token) return;

    // Build the SSE URL — JWT goes in query param because EventSource
    // doesn't support custom headers (browser limitation)
    const url = `${SSE_BASE}/stream/live?token=${encodeURIComponent(token)}`;

    // Open the SSE connection — browser keeps this HTTP connection open indefinitely
    const es = new EventSource(url);
    esRef.current = es;

    // Fires every time the gateway sends a "data: {...}\n\n" event.
    // Each event is one drone's StreamPayload for one frame.
    es.onmessage = (event: MessageEvent) => {
      try {
        // Parse the JSON string back into a StreamPayload object
        const payload = JSON.parse(event.data as string) as StreamPayload;

        // Update the Map — copy the previous map and overwrite just this drone's entry.
        // new Map(prev) is needed because React requires a new object reference to detect the change.
        setFrames((prev) => new Map(prev).set(payload.drone_id, payload));
      } catch {
        // Ignore malformed events — gateway keepalive comments never reach here
        // but any unexpected data won't crash the app
      }
    };

    es.onerror = () => {
      // EventSource auto-reconnects with backoff when the connection drops.
      // Token expiry is handled centrally in AuthContext (interval check every 60s).
    };

    // Cleanup: runs when the component unmounts or when token changes.
    // Closes the SSE connection so the gateway removes this client's queue
    // and stops sending frames to a disconnected client.
    return () => {
      es.close();
      esRef.current = null;
    };
  }, [token]); // Re-run if token changes (e.g. user logs out and back in)

  return frames;
}
