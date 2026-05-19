/*
 * SseBroadcaster.java — holds the list of all connected browser clients. When a frame
 * arrives it fans it out to every client. Also sends keepalive pings every 20 seconds
 * so proxies and browsers don't close idle connections.
 */
package com.uniview.gateway.sse;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.uniview.gateway.model.StreamPayload;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

@Component
public class SseBroadcaster {

    private static final Logger log = LoggerFactory.getLogger(SseBroadcaster.class);

    // One SseEmitter per connected browser tab.
    // CopyOnWriteArrayList: safe to iterate while subscribe() concurrently adds new emitters.
    // Writes (add/remove) are rare; reads (iteration during push) are frequent — ideal fit.
    private final List<SseEmitter> emitters = new CopyOnWriteArrayList<>();

    // Jackson's JSON serializer — converts StreamPayload to a JSON string once per frame,
    // then reuses that same string for every connected client.
    private final ObjectMapper objectMapper;

    public SseBroadcaster(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    // Called by StreamController every time the algorithm pushes a new frame.
    // Serializes the payload once, then sends it to every connected browser client.
    public void pushEvent(StreamPayload payload) {
        // Serialize once — cheaper than serializing separately for each client
        String json;
        try {
            // Converts the StreamPayload Java object into a JSON string
            json = objectMapper.writeValueAsString(payload);
        } catch (JsonProcessingException ex) {
            log.error("Failed to serialize StreamPayload for SSE", ex);
            return;
        }

        // Wrap the JSON string in an SSE event with content-type application/json
        SseEmitter.SseEventBuilder event =
                SseEmitter.event().data(json, MediaType.APPLICATION_JSON);

        // Collect dead emitters separately — modifying the list inside the loop is unsafe
        // even with CopyOnWriteArrayList. Remove them all after the loop finishes.
        List<SseEmitter> dead = new ArrayList<>();
        for (SseEmitter emitter : emitters) {
            try {
                emitter.send(event);
            } catch (IOException ex) {
                // IOException means this client disconnected — mark for removal
                dead.add(emitter);
            }
        }
        emitters.removeAll(dead);
    }

    // Called by StreamController when a browser hits GET /stream/live.
    // Creates a new SseEmitter, registers cleanup callbacks, and returns it to Spring.
    // Spring holds the HTTP response open and writes to it whenever emitter.send() is called.
    public SseEmitter subscribe() {
        // 0L = no framework timeout. We keep the connection alive with keepalive comments.
        // Default would be 30s which would close the stream during quiet periods.
        SseEmitter emitter = new SseEmitter(0L);

        emitters.add(emitter);
        log.info("SSE client subscribed. Total: {}", emitters.size());

        // Same cleanup lambda registered on all three exit paths:
        // onCompletion — browser closed the tab cleanly
        // onTimeout    — connection timed out (shouldn't happen with 0L, but defensive)
        // onError      — network dropped or connection reset
        Runnable cleanup = () -> {
            emitters.remove(emitter);
            log.info("SSE client unsubscribed. Total: {}", emitters.size());
        };

        // cases when the emitter is done and should be removed from the list of active emitters
        emitter.onCompletion(cleanup);
        emitter.onTimeout(cleanup);
        emitter.onError(ex -> cleanup.run());

        return emitter;
    }

    // Runs every 20 seconds after the previous run finishes (fixedDelay).
    // Sends an SSE comment line (": keepalive") to every client.
    // The browser ignores comments as data, but sending bytes resets the idle timer
    // on any proxy or load balancer — preventing it from closing silent connections.
    @Scheduled(fixedDelay = 20_000)
    public void sendKeepalive() {
        // No clients — no need to send keepalives
        if (emitters.isEmpty()) return;

        // SSE comment format: ": keepalive" — ignored by the browser as data
        SseEmitter.SseEventBuilder keepalive = SseEmitter.event().comment("keepalive");
        List<SseEmitter> dead = new ArrayList<>();

        for (SseEmitter emitter : emitters) {
            try {
                emitter.send(keepalive);
            } catch (IOException ex) {
                dead.add(emitter);
            }
        }
        emitters.removeAll(dead);
    }

    public int getClientCount() {
        return emitters.size();
    }
}
