import { apiRequest } from './client';
import type { LoginRequest, LoginResponse, CreateUserRequest, User } from '../../types/auth';

export const authAPI = {
  login: async (credentials: LoginRequest): Promise<LoginResponse> => {
    return apiRequest<LoginResponse>('/login', {
      method: 'POST',
      body: JSON.stringify(credentials),
    });
  },

  logout: async (): Promise<void> => {
    return apiRequest<void>('/logout', {
      method: 'POST',
    });
  },

  createUser: async (userData: CreateUserRequest): Promise<User> => {
    return apiRequest<User>('/users', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  },
};
