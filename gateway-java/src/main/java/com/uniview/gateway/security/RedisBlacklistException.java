/*
 * RedisBlacklistException.java — a custom exception type. Just a signal that says
 * "Redis failed" so JwtAuthFilter can catch it specifically and return 503.
 */
package com.uniview.gateway.security;

public class RedisBlacklistException extends RuntimeException {
    public RedisBlacklistException(String message, Throwable cause) {
        super(message, cause);
    }
}
