/*
 * AppConfig.java — creates the shared beans: WebClient (outbound HTTP), StringRedisTemplate
 * (Redis connection), CORS rules. Also enables the @Scheduled keepalive scheduler.
 */
package com.uniview.gateway.config;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.List;

@Configuration
@EnableScheduling
@EnableConfigurationProperties(AppProperties.class)
public class AppConfig {

    // springs http client
    @Bean
    public WebClient webClient() {
        return WebClient.builder()
                .codecs(configurer ->
                // sets the max response size the client will buffer in memory to 10MB.
                        configurer.defaultCodecs().maxInMemorySize(10 * 1024 * 1024))
                .build();
    }

    // CORS - Cross-Origin Resource Sharing
    @Bean
    public CorsConfigurationSource corsConfigurationSource(AppProperties props) {
        CorsConfiguration config = new CorsConfiguration();
        config.setAllowedOrigins(List.of(props.getFrontendUrl()));
        config.setAllowCredentials(true);
        config.setAllowedMethods(List.of("GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"));
        config.setAllowedHeaders(List.of("*"));
        config.setExposedHeaders(List.of("*"));

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        // apply this CORS config to every endpoint
        source.registerCorsConfiguration("/**", config);
        return source;
    }

    // redis
    // RedisConnectionFactory factory - opens redis connection using properties from application.properties (host, port)
    @Bean
    public StringRedisTemplate stringRedisTemplate(RedisConnectionFactory factory) {
        // remote control for Redis that speaks strings. add and remove tokens
        return new StringRedisTemplate(factory);
    }
}
