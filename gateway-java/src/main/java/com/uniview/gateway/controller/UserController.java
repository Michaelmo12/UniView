/*
 * UserController.java — GET/POST/DELETE /api/users/**. Admin user management proxied
 * to the Python backend. Non-admin users can only view their own profile.
 */
package com.uniview.gateway.controller;

import com.uniview.gateway.config.AppProperties;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import org.springframework.web.server.ResponseStatusException;

import java.time.Duration;
import java.util.Map;

@RestController
public class UserController {

    private final WebClient webClient;
    private final AppProperties props;

    public UserController(WebClient webClient, AppProperties props) {
        this.webClient = webClient;
        this.props = props;
    }

    // POST /api/users — admin only
    @PostMapping("/api/users")
    public ResponseEntity<?> addUser(@RequestBody Map<String, Object> userData,
                                     Authentication auth) {
        // if not admin throw 403.
        requireAdmin(auth);

        try {
            // create user in the database
            Map<?, ?> created = webClient.post()
                    .uri(props.getBackendUrl() + "/auth/add-user")
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(userData)
                    .retrieve()
                    .bodyToMono(Map.class)
                    .timeout(Duration.ofSeconds(10))
                    .block();
            return ResponseEntity.status(201).body(created);

        } catch (WebClientResponseException ex) {
            return ResponseEntity.status(ex.getStatusCode())
                    .body(Map.of("detail", ProxyUtils.extractDetail(ex)));
        } catch (Exception ex) {
            return ResponseEntity.status(503)
                    .body(Map.of("detail", "Backend service unavailable: " + ex.getMessage()));
        }
    }

    // GET /api/users/{id} — authenticated user can view own profile; admin can view any
    @GetMapping("/api/users/{id}")
    public ResponseEntity<?> getUserById(@PathVariable int id, Authentication auth) {
        Map<String, Object> currentUser = extractUser(auth);
        int currentUserId = ((Number) currentUser.get("user_id")).intValue();
        String role = (String) currentUser.getOrDefault("role", "");

        if (currentUserId != id && !"admin".equals(role)) {
            return ResponseEntity.status(403)
                    .body(Map.of("detail", "You can only view your own profile"));
        }

        try {
            Map<?, ?> user = webClient.get()
                    .uri(props.getBackendUrl() + "/users/" + id)
                    .retrieve()
                    .bodyToMono(Map.class)
                    .timeout(Duration.ofSeconds(5))
                    .block();
            return ResponseEntity.ok(user);

        } catch (WebClientResponseException ex) {
            if (ex.getStatusCode().value() == 404) {
                return ResponseEntity.status(404).body(Map.of("detail", "User not found"));
            }
            return ResponseEntity.status(ex.getStatusCode())
                    .body(Map.of("detail", ProxyUtils.extractDetail(ex)));
        } catch (Exception ex) {
            return ResponseEntity.status(503)
                    .body(Map.of("detail", "Backend service unavailable: " + ex.getMessage()));
        }
    }

    // GET /api/users/email/{email} — admin only
    // Spring MVC resolves literal path segments before path variables,
    // so this is matched before /api/users/{id} for the "email" segment.
    @GetMapping("/api/users/email/{email}")
    public ResponseEntity<?> getUserByEmail(@PathVariable String email, Authentication auth) {
        requireAdmin(auth);

        try {
            Map<?, ?> user = webClient.get()
                    .uri(props.getBackendUrl() + "/users/email/" + email)
                    .retrieve()
                    .bodyToMono(Map.class)
                    .timeout(Duration.ofSeconds(5))
                    .block();
            return ResponseEntity.ok(user);

        } catch (WebClientResponseException ex) {
            if (ex.getStatusCode().value() == 404) {
                return ResponseEntity.status(404).body(Map.of("detail", "User not found"));
            }
            return ResponseEntity.status(ex.getStatusCode())
                    .body(Map.of("detail", ProxyUtils.extractDetail(ex)));
        } catch (Exception ex) {
            return ResponseEntity.status(503)
                    .body(Map.of("detail", "Backend service unavailable: " + ex.getMessage()));
        }
    }

    // DELETE /api/users/{id} — admin only
    @DeleteMapping("/api/users/{id}")
    public ResponseEntity<?> deleteUser(@PathVariable int id, Authentication auth) {
        requireAdmin(auth);

        try {
            webClient.delete()
                    .uri(props.getBackendUrl() + "/users/" + id)
                    .retrieve()
                    .toBodilessEntity()
                    .timeout(Duration.ofSeconds(5))
                    .block();
            return ResponseEntity.noContent().build();

        } catch (WebClientResponseException ex) {
            if (ex.getStatusCode().value() == 404) {
                return ResponseEntity.status(404).body(Map.of("detail", "User not found"));
            }
            return ResponseEntity.status(ex.getStatusCode())
                    .body(Map.of("detail", ProxyUtils.extractDetail(ex)));
        } catch (Exception ex) {
            return ResponseEntity.status(503)
                    .body(Map.of("detail", "Backend service unavailable: " + ex.getMessage()));
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    @SuppressWarnings("unchecked")
    private Map<String, Object> extractUser(Authentication auth) {
        // JwtAuthFilter stores the verified user map as the principal
        return (Map<String, Object>) auth.getPrincipal();
    }

    private void requireAdmin(Authentication auth) {
        Map<String, Object> user = extractUser(auth);
        if (!"admin".equals(user.get("role"))) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Admin access required");
        }
    }
}
