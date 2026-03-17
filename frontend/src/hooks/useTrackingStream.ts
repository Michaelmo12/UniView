/**
 * useTrackingStream
 *
 * Custom hook managing a WebSocket connection to the gateway's /ws/stream endpoint.
 * Returns the most recent TrackingMessage parsed from the stream.
 *
 * Connects when token is available, disconnects on unmount.
 * JWT is passed as query param (browser WebSocket cannot set headers).
 */
import { useEffect, useRef, useState } from "react";
import type { TrackingMessage } from "../types/tracking";

export function useTrackingStream(
  url: string,
  token: string | null
): TrackingMessage | null {
  const [message, setMessage] = useState<TrackingMessage | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!token) return;

    const ws = new WebSocket(`${url}?token=${token}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("[TrackingStream] Connected to", url);
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data as string) as TrackingMessage;
        setMessage(data);
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onerror = (err) => {
      console.warn("[TrackingStream] WebSocket error:", err);
    };

    ws.onclose = () => {
      console.log("[TrackingStream] Connection closed");
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [url, token]);

  return message;
}
