/*
 * TokenBlacklist.java — knows how to write a token to Redis on logout and check if one
 * is blacklisted on every request. Pure Redis logic — no HTTP awareness.
 */

package com.uniview.gateway.security;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Duration;

@Service
public class TokenBlacklist {

    private static final Logger log = LoggerFactory.getLogger(TokenBlacklist.class);
    private static final String KEY_PREFIX = "blacklist:";

    private final StringRedisTemplate redis;

    public TokenBlacklist(StringRedisTemplate redis) {
        this.redis = redis;
    }

    // Stores token in Redis with TTL equal to the token's remaining lifetime.
    // When TTL expires Redis deletes the key automatically — no cleanup job needed.
    public void blacklist(String token, long remainingSeconds) {
        if (remainingSeconds <= 0) {
            return;
        }
        // .set(key, value, TTL) stores the key with a value and expiration time. We can just store "1" as the value since we only care about existence.
        redis.opsForValue().set(KEY_PREFIX + token, "1", Duration.ofSeconds(remainingSeconds));
    }

    // Returns true if the token is currently blacklisted.
    // Throws RedisBlacklistException on Redis error (fail-closed: caller returns 503).
    public boolean isBlacklisted(String token) {
        try {
            Boolean exists = redis.hasKey(KEY_PREFIX + token);
            // Redis returns null on error, treat as not blacklisted to avoid false positives, but log the error
            return Boolean.TRUE.equals(exists);
        } catch (Exception ex) {
            log.error("Redis error checking token blacklist", ex);
            throw new RedisBlacklistException("Unable to verify token blacklist status", ex);
        }
    }
}
