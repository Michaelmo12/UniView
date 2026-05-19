/*
 * TrackEntry.java — one detected person in a frame: bounding box (x, y, width, height),
 * global_id assigned by the re-identification algorithm, confidence score, and state.
 */
package com.uniview.gateway.model;

import com.fasterxml.jackson.annotation.JsonProperty;

public record TrackEntry(
        @JsonProperty("global_id")      int    globalId,
        int                                     x,
        int                                     y,
        int                                     width,
        int                                     height,
        double                                  confidence,
        String                                  state,
        @JsonProperty("frames_tracked") int    framesTracked
) {}
