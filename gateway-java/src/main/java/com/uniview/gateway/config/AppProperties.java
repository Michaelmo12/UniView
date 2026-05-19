/*
 * AppProperties.java — reads application.properties into Java objects on startup.
 * Every other class reads config from here instead of touching the properties file directly.
 */
package com.uniview.gateway.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

// find all properties prefixed with "app" in application.properties and bind to this class
@ConfigurationProperties(prefix = "app")
public class AppProperties {

    private final Jwt jwt = new Jwt();
    private String backendUrl;
    private String frontendUrl;

    public Jwt getJwt() {
        return jwt;
    }

    public String getBackendUrl() {
        return backendUrl;
    }

    public void setBackendUrl(String backendUrl) {
        this.backendUrl = backendUrl;
    }

    public String getFrontendUrl() {
        return frontendUrl;
    }

    public void setFrontendUrl(String frontendUrl) {
        this.frontendUrl = frontendUrl;
    }

    public static class Jwt {
        private String secretKey;
        private String algorithm;
        private int expirationMinutes;

        public String getSecretKey() {
            return secretKey;
        }

        public void setSecretKey(String secretKey) {
            this.secretKey = secretKey;
        }

        public String getAlgorithm() {
            return algorithm;
        }

        public void setAlgorithm(String algorithm) {
            this.algorithm = algorithm;
        }

        public int getExpirationMinutes() {
            return expirationMinutes;
        }

        public void setExpirationMinutes(int expirationMinutes) {
            this.expirationMinutes = expirationMinutes;
        }
    }
}
