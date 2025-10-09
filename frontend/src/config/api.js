// API Configuration
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? '' // Use relative URLs in production (same domain as frontend)
  : 'http://localhost:8001'; // Use localhost in development

export const API_ENDPOINTS = {
  PRODUCT_ANALYSIS: `${API_BASE_URL}/api/v1/product`,
  FEATURE_SEARCH: `${API_BASE_URL}/api/v1/features`,
  CHAT_ASSISTANT: `${API_BASE_URL}/api/v1/chat`,
  HEALTH: `${API_BASE_URL}/api/health`
};

export default API_BASE_URL;
