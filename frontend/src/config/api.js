// API Configuration
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'http://sentiment-analysis-alb-1018237225.us-east-1.elb.amazonaws.com' // Use ALB URL in production
  : 'http://localhost:8001'; // Use localhost in development

export const API_ENDPOINTS = {
  PRODUCT_ANALYSIS: `${API_BASE_URL}/api/v1/product`,
  FEATURE_SEARCH: `${API_BASE_URL}/api/v1/features`,
  CHAT_ASSISTANT: `${API_BASE_URL}/api/v1/chat`,
  HEALTH: `${API_BASE_URL}/api/health`
};

export default API_BASE_URL;
