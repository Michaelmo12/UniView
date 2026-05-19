/*
 * JwtUtil.java — knows how to create a JWT on login and decode/verify one on every request.
 * Pure JWT logic — no HTTP awareness. Everything JJWT-specific is contained here.
 */
package com.uniview.gateway.security;

import com.uniview.gateway.config.AppProperties;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.JwtException;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import org.springframework.stereotype.Component;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

@Component
public class JwtUtil {

    private final SecretKey signingKey;
    private final int expirationMinutes;

    public JwtUtil(AppProperties props) {
        this.signingKey = Keys.hmacShaKeyFor(
                props.getJwt().getSecretKey().getBytes(StandardCharsets.UTF_8));
        this.expirationMinutes = props.getJwt().getExpirationMinutes();
    }

    public String createJwt(int userId, String email, String role) {
        // issued at now, expires in expirationMinutes
        Instant now = Instant.now();
        // build JWT with user_id, email, role claims, signed with our secret key
        return Jwts.builder()
                .claim("user_id", userId)
                .claim("email", email)
                .claim("role", role)
                .issuedAt(Date.from(now))
                .expiration(Date.from(now.plus(expirationMinutes, ChronoUnit.MINUTES)))
                .signWith(signingKey)
                // compact the JWT to a string and return it
                .compact();
    }

    // Returns decoded claims. Throws JwtException on invalid, expired, or malformed token.
    public Claims extractClaims(String token) {
        return Jwts.parser()
                .verifyWith(signingKey)
                // parse the token and return the claims if valid, otherwise throw JwtException
                .build()
                .parseSignedClaims(token)
                .getPayload();
    }

    // Returns user map with user_id, email, role. Throws JwtException on any validation failure.
    public Map<String, Object> verifyJwt(String token) {
        Claims claims = extractClaims(token);

        Object userId = claims.get("user_id");
        String email  = claims.get("email", String.class);
        String role   = claims.get("role",  String.class);

        if (userId == null || email == null) {
            throw new JwtException("Missing required claims");
        }

        Map<String, Object> result = new HashMap<>();
        result.put("user_id", ((Number) userId).intValue());
        result.put("email", email);
        result.put("role", role);
        return result;
    }

    // Returns seconds remaining until token expires. Used by logout to set Redis TTL.
    public long getRemainingSeconds(String token) {
        Claims claims = extractClaims(token);
        Date exp = claims.getExpiration();
        // Calculate remaining seconds until expiration
        long remaining = (exp.getTime() - System.currentTimeMillis()) / 1000L;
        // If token is already expired, return 0 to avoid negative TTL in Redis
        return Math.max(0, remaining);
    }
}
