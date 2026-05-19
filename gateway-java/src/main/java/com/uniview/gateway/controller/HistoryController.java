/*
 * HistoryController.java — GET /api/history (proxied to Python backend) and
 * GET /api/algorithm/status (served from HistoryAggregator's in-memory snapshot).
 */
package com.uniview.gateway.controller;

import com.uniview.gateway.aggregator.HistoryAggregator;
import com.uniview.gateway.config.AppProperties;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import org.springframework.web.util.UriComponentsBuilder;

import java.time.Duration;
import java.util.List;
import java.util.Map;

@RestController
public class HistoryController {

    private final HistoryAggregator historyAggregator;
    private final WebClient webClient;
    private final AppProperties props;

    public HistoryController(HistoryAggregator historyAggregator,
                             WebClient webClient, AppProperties props) {
        this.historyAggregator = historyAggregator;
        this.webClient = webClient;
        this.props = props;
    }

    // Called by frontend Statistics page on mount to get current in-memory system snapshot.
    @GetMapping("/api/algorithm/status")
    public ResponseEntity<Map<String, Object>> getAlgorithmStatus() {
        return ResponseEntity.ok(historyAggregator.getCurrentStatus());
    }

    @GetMapping("/api/history")
    public ResponseEntity<?> getHistory(
            @RequestParam(name = "start_time", required = false) String startTime,
            @RequestParam(name = "end_time",   required = false) String endTime) {

        // Build the backend URL with optional query parameters.
        UriComponentsBuilder uriBuilder = UriComponentsBuilder
                .fromHttpUrl(props.getBackendUrl() + "/history/");
        if (startTime != null) uriBuilder.queryParam("start_time", startTime);
        if (endTime   != null) uriBuilder.queryParam("end_time",   endTime);

        try {
            List<?> body = webClient.get()
                    .uri(uriBuilder.build().toUri())
                    .retrieve()
                    .bodyToMono(List.class)
                    .timeout(Duration.ofSeconds(5))
                    .block();
            return ResponseEntity.ok(body);

        } catch (WebClientResponseException ex) {
            return ResponseEntity.status(ex.getStatusCode())
                    .body(Map.of("detail", ProxyUtils.extractDetail(ex)));

        } catch (Exception ex) {
            return ResponseEntity.status(503)
                    .body(Map.of("detail", "Backend service unavailable: " + ex.getMessage()));
        }
    }
}
