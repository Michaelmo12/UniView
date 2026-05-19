/*
 * GatewayApplication.java — entry point. The main() method that boots the entire
 * Spring Boot application. One line effectively starts Tomcat, loads all beans,
 * and opens port 8080.
 */
package com.uniview.gateway;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class GatewayApplication {
    public static void main(String[] args) {
        SpringApplication.run(GatewayApplication.class, args);
    }
}
