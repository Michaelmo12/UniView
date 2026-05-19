/*
 * HistoryAggregator.java — receives every pushed frame, groups them by minute, and when
 * the minute changes flushes aggregated stats (avg people, peak people, active drones)
 * to the Python backend for storage. Also serves the live status snapshot to the frontend.
 */
package com.uniview.gateway.aggregator;

import com.uniview.gateway.config.AppProperties;
import com.uniview.gateway.model.StreamPayload;
import com.uniview.gateway.model.TrackEntry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicLong;

@Service
public class HistoryAggregator {

    private static final Logger log = LoggerFactory.getLogger(HistoryAggregator.class);

    // minute_key (e.g. "2026-03-26T14:05") → all payloads received in that minute
    private final Map<String, List<StreamPayload>> buffers = new ConcurrentHashMap<>();

    // volatile so all threads see the latest value after a write without caching issues
    private volatile String currentMinute = null;

    // AtomicLong is thread-safe version of long, used to detect whether the pipeline is still online
    private final AtomicLong lastPayloadAt = new AtomicLong(0);

    private final WebClient webClient;
    private final AppProperties props;

    // Constructor injection — Spring provides the WebClient and AppProperties beans
    public HistoryAggregator(WebClient webClient, AppProperties props) {
        this.webClient = webClient;
        this.props = props;
    }

    // Buffer an incoming payload by its truncated minute key.
    // When the minute rolls over, flush the completed minute to the backend.
    public void processPayload(StreamPayload payload) {
        // "2026-03-26T14:05:47" → "2026-03-26T14:05" — drop the seconds
        String minuteKey = payload.timestamp().substring(0, 16);

        // First payload ever — initialize currentMinute
        if (this.currentMinute == null) {
            this.currentMinute = minuteKey;
        }

        // Minute rolled over — flush the completed minute, then start tracking the new one
        if (!minuteKey.equals(this.currentMinute)) {
            flush(this.currentMinute);
            this.buffers.remove(this.currentMinute);
            this.currentMinute = minuteKey;
        }

        // Add payload to this minute's buffer. computeIfAbsent creates the list if missing.
        this.buffers.computeIfAbsent(minuteKey, k -> new CopyOnWriteArrayList<>()).add(payload);
        this.lastPayloadAt.set(System.nanoTime());
    }

    // Compute per-minute aggregates and POST them to the backend.
    // Failures are logged as warnings — never propagated to callers.
    private void flush(String minuteKey) {
        
        // get the list for this minute key, but if there's no entry for it, give me an empty list instead of null.
        List<StreamPayload> payloads = this.buffers.getOrDefault(minuteKey, List.of());
        
        if (payloads.isEmpty()) {
            log.warn("flush called for minute {} but buffer is empty", minuteKey);
            return;
        }

        // Count tracks per frame so we can compute average and peak
        List<Integer> trackCounts = new ArrayList<>();
        for (StreamPayload p : payloads) {
            trackCounts.add(p.tracks().size());
        }

        int sum = 0;
        for (int c : trackCounts) sum += c;
        int avgPeople  = (int) Math.round((double) sum / trackCounts.size());
        int peakPeople = Collections.max(trackCounts);

        // Take the max reported value across all frames for drones and reid matches
        int activeDrones = 0;
        int totalReid    = 0;
        for (StreamPayload p : payloads) {
            activeDrones = Math.max(activeDrones, p.activeDronesCount());
            totalReid    = Math.max(totalReid,    p.totalReidMatches());
        }

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("timestamp",           minuteKey + ":00+00:00");
        body.put("avg_people_count",    avgPeople);
        body.put("peak_people_count",   peakPeople);
        body.put("active_drones_count", activeDrones);
        body.put("total_reid_matches",  totalReid);

        int payloadCount = payloads.size();

        // Fire-and-forget: .subscribe() starts the HTTP call on a background thread.
        // processPayload() returns immediately without waiting — backend failures only log a warning.
        this.webClient.post()
                .uri(this.props.getBackendUrl() + "/history/ingest")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .toBodilessEntity()
                .timeout(Duration.ofSeconds(5))
                .subscribe(
                        resp -> log.info("Flushed history for minute {} ({} payloads)",
                                minuteKey, payloadCount),
                        err  -> log.warn("Failed to flush history for minute {}: {}",
                                minuteKey, err.getMessage())
                );
    }

    // Return a real-time snapshot of the current minute's buffer for the statistics page.
    public Map<String, Object> getCurrentStatus() {
        // Pipeline is online if a payload arrived within the last 10 seconds (10_000_000_000 ns)
        boolean pipelineOnline = this.lastPayloadAt.get() != 0
                && (System.nanoTime() - this.lastPayloadAt.get()) < 10_000_000_000L;

        // Default response when no data has arrived yet
        Map<String, Object> defaultStatus = new LinkedHashMap<>();
        defaultStatus.put("active_drones",   0);
        defaultStatus.put("active_tracks",   0);
        defaultStatus.put("server_fps",      0.0);
        defaultStatus.put("system_status",   "Optimal");
        defaultStatus.put("pipeline_online", pipelineOnline);

        if (this.currentMinute == null) return defaultStatus;

        List<StreamPayload> payloads = this.buffers.getOrDefault(this.currentMinute, List.of());
        // If no payloads for the current minute, return defaults with pipelineOnline updated based on lastPayloadAt
        if (payloads.isEmpty()) return defaultStatus;

        // Max drones reported across all payloads this minute
        int activeDrones = 0;
        for (StreamPayload p : payloads) {
            activeDrones = Math.max(activeDrones, p.activeDronesCount());
        }

        // Unique global_ids from the latest payload per drone (avoids double-counting same person)
        Map<String, StreamPayload> latestPerDrone = new LinkedHashMap<>();
        for (StreamPayload p : payloads) {
            latestPerDrone.put(p.droneId(), p);
        }

        Set<Integer> uniqueGlobalIds = new HashSet<>();
        for (StreamPayload p : latestPerDrone.values()) {
            for (TrackEntry track : p.tracks()) {
                uniqueGlobalIds.add(track.globalId());
            }
        }
        // gives the number of unique people.
        int activeTracks = uniqueGlobalIds.size();

        // Average confidence across payloads that have a non-zero value
        List<Double> confidences = new ArrayList<>();
        for (StreamPayload p : payloads) {
            if (p.avgConfidence() > 0.0) confidences.add(p.avgConfidence());
        }
        double avgConfidence = 0.0;
        if (!confidences.isEmpty()) {
            double csum = 0;
            for (double c : confidences) csum += c;
            avgConfidence = Math.round((csum / confidences.size()) * 1000.0) / 1000.0;
        }

        // Average pipeline latency across all payloads this minute
        double latencySum = 0;
        for (StreamPayload p : payloads) latencySum += p.pipelineLatencyMs();
        double avgLatency = latencySum / payloads.size();

        // Convert avg latency (ms/frame) → FPS. One decimal place.
        double serverFps = avgLatency > 0
                ? Math.round(1000.0 / avgLatency * 10.0) / 10.0
                : 0.0;

        // Status thresholds: <1000ms = Optimal, <2000ms = Warning, else Critical
        String systemStatus;
        if      (avgLatency < 1000) systemStatus = "Optimal";
        else if (avgLatency < 2000) systemStatus = "Warning";
        else                        systemStatus = "Critical";

        // Average per-stage timings across payloads that include them
        String[] stageKeys = {"detection", "features", "fusion", "reconstruction", "tracking", "total"};
        Map<String, List<Double>> stageSamples = new LinkedHashMap<>();
        // for every key init an array
        for (String k : stageKeys) stageSamples.put(k, new ArrayList<>());

        //For every payload, read its stage timings map and append each value to the matching sample list. containsKey skips stages that weren't reported in that payload.
        for (StreamPayload p : payloads) {
            Map<String, Double> timings = p.stageTimingsMs();
            for (String k : stageKeys) {
                if (timings.containsKey(k)) stageSamples.get(k).add(timings.get(k));
            }
        }

        Map<String, Double> avgStageTimings = new LinkedHashMap<>();
        for (String k : stageKeys) {
            List<Double> samples = stageSamples.get(k);
            if (samples.isEmpty()) {
                avgStageTimings.put(k, 0.0);
            } else {
                double tsum = 0;
                for (double v : samples) tsum += v;
                avgStageTimings.put(k, Math.round(tsum / samples.size() * 10.0) / 10.0);
            }
        }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("active_drones",           activeDrones);
        result.put("active_tracks",           activeTracks);
        result.put("server_fps",              serverFps);
        result.put("system_status",           systemStatus);
        result.put("avg_pipeline_latency_ms", Math.round(avgLatency * 10.0) / 10.0);
        result.put("avg_confidence",          avgConfidence);
        result.put("stage_timings_ms",        avgStageTimings);
        result.put("pipeline_online",         pipelineOnline);
        return result;
    }
}
