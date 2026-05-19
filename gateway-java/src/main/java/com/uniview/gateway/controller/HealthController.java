/*
 * HealthController.java — GET /health (always returns pass) and GET /health/ready
 * (pings the Python backend; returns 503 if it is unreachable).
 */
package com.uniview.gateway.controller;

import com.uniview.gateway.config.AppProperties;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;

@RestController
public class HealthController {

    private final WebClient webClient;
    private final AppProperties props;

    public HealthController(WebClient webClient, AppProperties props) {
        this.webClient = webClient;
        this.props = props;
    }

    // GET /health is a simple liveness check that always returns pass.
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("status",    "pass");
        body.put("service",   "gateway");
        body.put("version",   "1.0.0");
        body.put("timestamp", Instant.now().toString());
        return ResponseEntity.ok(body);
    }

    // GET /health/ready checks if the backend is reachable. If not, returns 503 with details.
    @GetMapping("/health/ready")
    public ResponseEntity<Map<String, Object>> ready() {
        Map<String, Object> healthStatus = new LinkedHashMap<>();
        healthStatus.put("status",    "pass");
        healthStatus.put("service",   "gateway");
        healthStatus.put("timestamp", Instant.now().toString());

        Map<String, Object> checks = new LinkedHashMap<>();
        boolean allHealthy = true;

        long start = System.currentTimeMillis();
        try {
            webClient.get()
                    .uri(props.getBackendUrl() + "/health")
                    .retrieve()
                    .toBodilessEntity()
                    .timeout(Duration.ofSeconds(5))
                    .block();

            long elapsed = System.currentTimeMillis() - start;
            checks.put("backend", Map.of(
                    "status",           "pass",
                    "response_time_ms", elapsed,
                    "url",              props.getBackendUrl()
            ));

        } catch (Exception ex) {
            allHealthy = false;
            checks.put("backend", Map.of(
                    "status", "fail",
                    "error",  ex.getMessage() != null ? ex.getMessage() : "unknown",
                    "url",    props.getBackendUrl()
            ));
        }

        healthStatus.put("checks", checks);

        if (!allHealthy) {
            healthStatus.put("status", "fail");
            return ResponseEntity.status(503).body(healthStatus);
        }
        return ResponseEntity.ok(healthStatus);
    }
}
