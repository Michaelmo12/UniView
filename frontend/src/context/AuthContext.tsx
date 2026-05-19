import { createContext, useContext, useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import { authAPI } from '../services/api/auth';
import type { User, AuthContextType } from '../types/auth';

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    // Restore user profile from storage on page load.
    // The JWT lives in an HttpOnly cookie — the browser sends it automatically.
    // We only need to restore the user object for the UI.
    const rememberMe = localStorage.getItem('rememberMe') === 'true';
    const storage = rememberMe ? localStorage : sessionStorage;
    const storedUser = storage.getItem('user');
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser));
      } catch {
        storage.removeItem('user');
      }
    }
  }, []);

  const login = async (email: string, password: string, rememberMe: boolean = false) => {
    const response = await authAPI.login({ email, password });

    setUser(response.user);

    // Store rememberMe preference and user profile for UI restoration on reload.
    // The JWT itself is stored in an HttpOnly cookie set by the gateway.
    localStorage.setItem('rememberMe', rememberMe.toString());
    const storage = rememberMe ? localStorage : sessionStorage;
    storage.setItem('user', JSON.stringify(response.user));
  };

  const logout = async () => {
    try {
      await authAPI.logout();
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setUser(null);
      localStorage.removeItem('user');
      localStorage.removeItem('rememberMe');
      sessionStorage.removeItem('user');
      sessionStorage.removeItem('welcome_shown');
    }
  };

  const value: AuthContextType = {
    user,
    login,
    logout,
    isAuthenticated: !!user,
    isAdmin: user?.role === 'admin',
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
