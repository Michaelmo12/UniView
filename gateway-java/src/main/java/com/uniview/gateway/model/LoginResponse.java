/*
 * LoginResponse.java — the user object sent back to the frontend after successful login.
 * The JWT itself is not in the body — it is set as an HttpOnly cookie by AuthController.
 */
package com.uniview.gateway.model;

import java.util.Map;

public record LoginResponse(
        Map<String, Object> user
) {}
