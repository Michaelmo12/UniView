/*
 * AuthController.java — POST /api/login and POST /api/logout. Login proxies credentials
 * to the Python backend, creates a JWT, and sets it as an HttpOnly cookie. Logout
 * blacklists the token in Redis and clears the cookie.
 */
package com.uniview.gateway.controller;

import com.uniview.gateway.config.AppProperties;
import com.uniview.gateway.model.LoginRequest;
import com.uniview.gateway.model.LoginResponse;
import com.uniview.gateway.security.JwtUtil;
import com.uniview.gateway.security.TokenBlacklist;
import io.jsonwebtoken.JwtException;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseCookie;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.Arrays;
import java.util.Map;

@RestController
public class AuthController {

    private final WebClient webClient;
    private final JwtUtil jwtUtil;
    private final TokenBlacklist tokenBlacklist;
    private final AppProperties props;

    public AuthController(WebClient webClient, JwtUtil jwtUtil,
                          TokenBlacklist tokenBlacklist, AppProperties props) {
        this.webClient = webClient;
        this.jwtUtil = jwtUtil;
        this.tokenBlacklist = tokenBlacklist;
        this.props = props;
    }

    @PostMapping("/api/login")
    public ResponseEntity<?> login(@RequestBody LoginRequest request,
                                   HttpServletResponse response) {
        // email and password from frontend
        Map<String, String> backendBody = Map.of(
                "email",    request.email(),
                "password", request.password()
        );


        try {
            // http call to backend /auth/login, which returns user info if successful
            // else throws WebClientResponseException with status 401 or 500
            Map<String, Object> user = webClient.post()
                    .uri(props.getBackendUrl() + "/auth/login")
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(backendBody)
                    .retrieve()
                    .bodyToMono(new ParameterizedTypeReference<Map<String, Object>>() {})
                    .timeout(Duration.ofSeconds(10))
                    .block();

            int    userId = ((Number) user.get("id")).intValue();
            String email  = (String) user.get("email");
            String role   = (String) user.get("role");

            String token = jwtUtil.createJwt(userId, email, role);

            // Set JWT as HttpOnly cookie — JS cannot read it, so XSS cannot steal it.
            // SameSite=Lax blocks CSRF from cross-site navigation while allowing
            // same-site requests. Secure=false for localhost development (HTTP only).
            ResponseCookie cookie = ResponseCookie.from("token", token)
                    .httpOnly(true)
                    .secure(false)
                    .path("/")
                    .maxAge(Duration.ofMinutes(props.getJwt().getExpirationMinutes()))
                    .sameSite("Lax")
                    .build();
            response.addHeader(HttpHeaders.SET_COOKIE, cookie.toString());

            // return HTTP 200 with user info (except password) in body
            return ResponseEntity.ok(new LoginResponse(user));

        } catch (WebClientResponseException ex) {
            if (ex.getStatusCode().value() == 401) {
                return ResponseEntity.status(401)
                        .body(Map.of("detail", "Incorrect email or password"));
            }
            return ResponseEntity.status(ex.getStatusCode())
                    .body(Map.of("detail", ProxyUtils.extractDetail(ex)));

        } catch (Exception ex) {
            return ResponseEntity.status(503)
                    .body(Map.of("detail", "Backend service unavailable: " + ex.getMessage()));
        }
    }

    @PostMapping("/api/logout")
    public ResponseEntity<?> logout(HttpServletRequest request,
                                    HttpServletResponse response) {
        // Read token from HttpOnly cookie
        String token = null;
        if (request.getCookies() != null) {
            token = Arrays.stream(request.getCookies())
                    .filter(c -> "token".equals(c.getName()))
                    .map(Cookie::getValue)
                    .findFirst()
                    .orElse(null);
        }

        // If no token cookie, return 401 Unauthorized
        if (token == null) {
            return ResponseEntity.status(401).body(Map.of("detail", "Missing token"));
        }

        // Verify token and extract user info for response.
        // If token is invalid, return 401.
        Map<String, Object> userMap;
        try {
            userMap = jwtUtil.verifyJwt(token);
        } catch (JwtException ex) {
            return ResponseEntity.status(401).body(Map.of("detail", "Invalid token"));
        }

        // Blacklist the token in Redis
        // with expiration equal to remaining JWT lifetime
        try {
            long remaining = jwtUtil.getRemainingSeconds(token);
            tokenBlacklist.blacklist(token, remaining);
        } catch (Exception ex) {
            return ResponseEntity.status(500)
                    .body(Map.of("detail", "Logout failed: " + ex.getMessage()));
        }

        // Clear the cookie by setting it with maxAge=0
        ResponseCookie clearCookie = ResponseCookie.from("token", "")
                .httpOnly(true)
                .secure(false)
                .path("/")
                .maxAge(0)
                .sameSite("Lax")
                .build();
        response.addHeader(HttpHeaders.SET_COOKIE, clearCookie.toString());

        // Return success message with user email from token
        return ResponseEntity.ok(Map.of(
                "message", "Successfully logged out",
                "user",    userMap.get("email")
        ));
    }
}
