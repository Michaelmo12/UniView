import { createContext, useContext, useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import { authAPI } from '../services/api/auth';
import type { User, AuthContextType } from '../types/auth';

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);

  useEffect(() => {
    // Check localStorage first (rememberMe = true), then sessionStorage (rememberMe = false)
    const rememberMe = localStorage.getItem('rememberMe') === 'true';
    const storage = rememberMe ? localStorage : sessionStorage;

    const storedToken = storage.getItem('token');
    const storedUser = storage.getItem('user');

    if (storedToken && storedUser) {
      setToken(storedToken);
      setUser(JSON.parse(storedUser));
    }
  }, []);

  const login = async (email: string, password: string, rememberMe: boolean = false) => {
    const response = await authAPI.login({ email, password });

    setToken(response.access_token);
    setUser(response.user);

    // Store rememberMe preference in localStorage (persists across sessions)
    localStorage.setItem('rememberMe', rememberMe.toString());

    // Store token/user in appropriate storage based on rememberMe
    const storage = rememberMe ? localStorage : sessionStorage;
    storage.setItem('token', response.access_token);
    storage.setItem('user', JSON.stringify(response.user));
  };

  const logout = async () => {
    try {
      await authAPI.logout();
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setToken(null);
      setUser(null);
      // Clear from both storages to ensure complete logout
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      sessionStorage.removeItem('token');
      sessionStorage.removeItem('user');
    }
  };

  const value: AuthContextType = {
    user,
    token,
    login,
    logout,
    isAuthenticated: !!token && !!user,
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
