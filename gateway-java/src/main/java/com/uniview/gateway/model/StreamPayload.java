/*
 * StreamPayload.java — one frame from one drone: the JPEG image, list of detected tracks,
 * timestamp, drone ID, and pipeline performance metrics. Received from the algorithm,
 * fanned out to SSE clients, and buffered by HistoryAggregator.
 */
package com.uniview.gateway.model;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

public record StreamPayload(
        String                                       timestamp,
        @JsonProperty("drone_id")            String  droneId,
        @JsonProperty("frame_base64")        String  frameBase64,
        List<TrackEntry>                             tracks,
        @JsonProperty("active_drones_count") int     activeDronesCount,
        @JsonProperty("total_reid_matches")  int     totalReidMatches,
        @JsonProperty("pipeline_latency_ms") double  pipelineLatencyMs,
        @JsonProperty("avg_confidence")      double  avgConfidence,
        @JsonProperty("stage_timings_ms")    Map<String, Double> stageTimingsMs
) {
    // Default stageTimingsMs to empty map when absent from JSON
    public StreamPayload {
        if (stageTimingsMs == null) stageTimingsMs = Map.of();
    }
}
