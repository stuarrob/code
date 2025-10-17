/**
 * API client for ETFTrader backend
 */

import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Portfolio APIs
export const portfolioAPI = {
  list: () => api.get('/portfolios'),
  get: (id: number) => api.get(`/portfolios/${id}`),
  create: (data: any) => api.post('/portfolios', data),
  update: (id: number, data: any) => api.patch(`/portfolios/${id}`, data),
  delete: (id: number) => api.delete(`/portfolios/${id}`),
};

// Position APIs
export const positionAPI = {
  list: (portfolioId: number) => api.get(`/positions/${portfolioId}/positions`),
  getDrift: (portfolioId: number) => api.get(`/positions/${portfolioId}/drift`),
};

// Trade APIs
export const tradeAPI = {
  list: (portfolioId: number) => api.get(`/trades/${portfolioId}/trades`),
  execute: (data: any) => api.post('/trades', data),
};

// Factor APIs
export const factorAPI = {
  latest: () => api.get('/factors/latest'),
  recommendations: (numPositions: number = 20) =>
    api.get(`/factors/recommendations?num_positions=${numPositions}`),
};

// Performance APIs
export const performanceAPI = {
  get: (portfolioId: number) => api.get(`/performance/${portfolioId}`),
  attribution: (portfolioId: number) => api.get(`/performance/${portfolioId}/attribution`),
};

// Risk APIs
export const riskAPI = {
  get: (portfolioId: number) => api.get(`/risk/${portfolioId}`),
  vix: () => api.get('/risk/vix'),
};

// Data APIs
export const dataAPI = {
  status: () => api.get('/data/status'),
};

export default api;
