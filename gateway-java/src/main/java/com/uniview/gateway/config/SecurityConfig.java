/*
 * SecurityConfig.java — defines the security filter chain: which endpoints are public,
 * which require auth, CORS policy, CSRF disabled, sessions stateless, and filter order.
 */
package com.uniview.gateway.config;

import com.uniview.gateway.security.JwtAuthFilter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsConfigurationSource;

@Configuration
// activates Spring Security. Without this, no request filtering happens at all.
@EnableWebSecurity
public class SecurityConfig {

    private final JwtAuthFilter jwtAuthFilter;
    private final CorsConfigurationSource corsConfigurationSource;

    // spring will automatically inject the JwtAuthFilter and CorsConfigurationSource beans we defined in AppConfig
    public SecurityConfig(JwtAuthFilter jwtAuthFilter,
                          CorsConfigurationSource corsConfigurationSource) {
        //custom filter that checks for JWT in the Authorization header and validates it. If valid, sets the user in the security context
        this.jwtAuthFilter = jwtAuthFilter;
        this.corsConfigurationSource = corsConfigurationSource;
    }

    // ====== security filter chain ======
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .csrf(AbstractHttpConfigurer::disable)
            .cors(cors -> cors.configurationSource(corsConfigurationSource))
            .sessionManagement(sm ->
                    sm.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .authorizeHttpRequests(auth -> auth
                    // Allow all CORS preflight requests before any auth check
                    .requestMatchers(HttpMethod.OPTIONS, "/**").permitAll()
                    .requestMatchers(
                            "/api/internal/push",
                            "/api/login",
                            "/health",
                            "/health/ready"
                    ).permitAll()
                    .anyRequest().authenticated()
            )
            .addFilterBefore(jwtAuthFilter, UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }
}
