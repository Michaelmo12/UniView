/*
 * JwtAuthFilter.java — the gatekeeper. Runs on every protected request. Reads the cookie,
 * calls TokenBlacklist to check if revoked, calls JwtUtil to verify the signature, then
 * puts the user into the SecurityContext so controllers can see who is logged in.
 */
package com.uniview.gateway.security;

import io.jsonwebtoken.JwtException;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.Cookie;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

// OncePerRequestFilter guarantees doFilterInternal only fires once per real HTTP request.
// Without this, Spring could call the filter multiple times for internal forwards/errors.
@Component
public class JwtAuthFilter extends OncePerRequestFilter {

    private final JwtUtil jwtUtil;
    private final TokenBlacklist tokenBlacklist;

    public JwtAuthFilter(JwtUtil jwtUtil, TokenBlacklist tokenBlacklist) {
        this.jwtUtil = jwtUtil;
        this.tokenBlacklist = tokenBlacklist;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain filterChain)
            throws ServletException, IOException {

        // ── Step 0: extract JWT from the HttpOnly cookie named "token" ──────────────
        // getCookies() returns null if the browser sent no cookies at all
        String token = null;
        if (request.getCookies() != null) {
            token = Arrays.stream(request.getCookies())
                    // keep only the cookie whose name is "token"
                    .filter(c -> "token".equals(c.getName()))
                    // extract the JWT string value from the matching cookie
                    .map(Cookie::getValue)
                    // take the first match, or null if no "token" cookie exists
                    .findFirst()
                    .orElse(null);
        }

        // No cookie → pass through unauthenticated.
        // SecurityConfig's anyRequest().authenticated() will reject the request with 401.
        if (token == null) {
            filterChain.doFilter(request, response);
            return;
        }

        // ── Step 1: blacklist check ───────────────────────────────────────────────
        // Fail-closed: if Redis is down we return 503 rather than letting the request through.
        // A logged-out user must never be allowed in just because Redis is temporarily unavailable.
        try {
            if (tokenBlacklist.isBlacklisted(token)) {
                // 401 Unauthorized — token was revoked on logout
                response.sendError(HttpServletResponse.SC_UNAUTHORIZED, "Token has been revoked");
                return;
            }
        } catch (RedisBlacklistException ex) {
            // 503 Service Unavailable — Redis is down, cannot verify blacklist
            response.sendError(HttpServletResponse.SC_SERVICE_UNAVAILABLE,
                    "Authentication service temporarily unavailable");
            return;
        }

        // ── Step 2: JWT signature + claims validation ─────────────────────────────
        // verifyJwt() checks signature, expiry, and required claims.
        // Throws JwtException on any failure (expired, tampered, malformed).
        try {
            Map<String, Object> userMap = jwtUtil.verifyJwt(token);

            // Build a Spring Security authority string from the role claim, e.g. "ROLE_ADMIN"
            String role = (String) userMap.getOrDefault("role", "user");
            List<SimpleGrantedAuthority> authorities =
                    List.of(new SimpleGrantedAuthority("ROLE_" + role.toUpperCase()));

            // Store the verified user map as the principal so controllers can read it via
            // auth.getPrincipal(). The token string is stored as the credentials field.
            UsernamePasswordAuthenticationToken auth =
                    new UsernamePasswordAuthenticationToken(userMap, token, authorities);
            auth.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));

            // Put the authenticated user into the thread-local SecurityContext.
            // Every class running on this thread can now call
            // SecurityContextHolder.getContext().getAuthentication() to get the user.
            SecurityContextHolder.getContext().setAuthentication(auth);

        } catch (JwtException ex) {
            // 401 Unauthorized — token signature invalid, expired, or malformed
            response.sendError(HttpServletResponse.SC_UNAUTHORIZED, "Could not validate credentials");
            return;
        }

        // ── Step 3: pass the now-authenticated request to the next filter / controller
        filterChain.doFilter(request, response);
    }

    // Called by Spring before doFilterInternal. Return true to skip this filter entirely.
    // These endpoints are public — no JWT required.
    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        String path = request.getServletPath();
        return path.equals("/api/internal/push")
                || path.equals("/api/login")
                || path.equals("/health")
                || path.equals("/health/ready");
    }
}
