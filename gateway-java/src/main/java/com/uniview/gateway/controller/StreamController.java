/*
 * StreamController.java — POST /api/internal/push (algorithm sends frames here) and
 * GET /stream/live (browser connects here for SSE). Auth on /stream/live is handled
 * by JwtAuthFilter via the HttpOnly cookie — no manual JWT check needed here.
 */
package com.uniview.gateway.controller;

import com.uniview.gateway.aggregator.HistoryAggregator;
import com.uniview.gateway.model.StreamPayload;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import com.uniview.gateway.sse.SseBroadcaster;

import java.util.Map;

@RestController
public class StreamController {
    // SseBroadcaster manages all SSE connections and broadcasting events to them.
    private final SseBroadcaster broadcaster;
    // HistoryAggregator processes incoming StreamPayloads and maintains recent history for new clients.
    private final HistoryAggregator historyAggregator;

    // Constructor injection of dependencies
    public StreamController(SseBroadcaster broadcaster, HistoryAggregator historyAggregator) {
        this.broadcaster = broadcaster;
        this.historyAggregator = historyAggregator;
    }

    // Internal endpoint: algorithm pushes StreamPayload here. No auth — internal network only.
    @PostMapping("/api/internal/push")
    public ResponseEntity<Map<String, String>> push(@RequestBody StreamPayload payload) {
        broadcaster.pushEvent(payload);
        historyAggregator.processPayload(payload);
        return ResponseEntity.ok(Map.of("status", "ok"));
    }

    // SSE stream endpoint. Auth is handled by JwtAuthFilter via the HttpOnly cookie —
    // no manual JWT check needed here. EventSource sends cookies automatically.
    @GetMapping(value = "/stream/live", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamLive(HttpServletResponse response) {
        // Required SSE headers so proxies don't buffer or cache the stream
        response.setHeader("Cache-Control", "no-cache");
        // Disable buffering in Nginx with X-Accel-Buffering: no
        response.setHeader("X-Accel-Buffering", "no");

        return broadcaster.subscribe();
    }
}
