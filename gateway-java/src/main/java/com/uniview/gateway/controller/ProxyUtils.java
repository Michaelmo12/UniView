/*
 * ProxyUtils.java — shared helper used by all proxy controllers. Extracts the error
 * message from a failed backend response so the gateway can forward it cleanly.
 */
package com.uniview.gateway.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import java.util.Map;

// Shared helper for all proxy controllers.
// Tries to parse {"detail": "..."} from backend JSON body, falls back gracefully.
final class ProxyUtils {
    private static final ObjectMapper MAPPER = new ObjectMapper();
    // just helper never create an object
    private ProxyUtils() {}

    static String extractDetail(WebClientResponseException ex) {
        try {
            String body = ex.getResponseBodyAsString();
            Map<?, ?> json = MAPPER.readValue(body, Map.class);
            Object detail = json.get("detail");
            if (detail instanceof String s) return s;
            return body;
        } catch (Exception ignored) {
            return "Backend error (status " + ex.getStatusCode().value() + ")";
        }
    }
}
