const API_BASE_URL = import.meta.env.VITE_API_GATEWAY_URL || 'http://localhost:8080/api';

class APIError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'APIError';
    this.status = status;
  }
}

// Called on any 401 response — clears user state and redirects to login.
export function handleUnauthorized(): void {
  localStorage.removeItem('user');
  localStorage.removeItem('rememberMe');
  sessionStorage.removeItem('user');
  window.location.href = '/login';
}

export async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  };

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    // credentials: 'include' sends the HttpOnly cookie with every request automatically
    credentials: 'include',
    headers,
  });

  if (!response.ok) {
    // Token expired or revoked — clear session and redirect to login.
    // Skip on /login itself: a 401 there means wrong credentials, not an expired session.
    if (response.status === 401 && !endpoint.includes('/login')) {
      handleUnauthorized();
    }
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new APIError(response.status, error.detail || `HTTP ${response.status}`);
  }

  if (response.status === 204) {
    return {} as T;
  }

  return response.json();
}
