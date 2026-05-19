/*
 * LoginRequest.java — email and password coming from the frontend login form.
 * Deserialized from the request body and forwarded to the Python backend.
 */
package com.uniview.gateway.model;

public record LoginRequest(String email, String password) {}
