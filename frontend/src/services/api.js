import axios from 'axios';

// Your Python backend URL
const API_URL = 'http://localhost:8000';

// API functions
export const chatbotAPI = {
  // Send message to chatbot
  sendMessage: async (message) => {
    try {
      const response = await axios.post(`${API_URL}/predict`, {
        text: message,
        conversation_id: `session_${Date.now()}`
      });
      
      return {
        success: true,
        data: response.data
        // response.data contains:
        // - intent: string
        // - confidence: number (0-1)
        // - response: string (bot's answer)
        // - timestamp: string
      };
    } catch (error) {
      console.error('API Error:', error);
      return {
        success: false,
        error: error.message
      };
    }
  },

  // Optional: Check if backend is running
  checkConnection: async () => {
    try {
      const response = await axios.get(`${API_URL}/health`);
      return response.data.status === 'healthy';
    } catch (error) {
      return false;
    }
  },

  // Optional: Get analytics
  getAnalytics: async () => {
    try {
      const response = await axios.get(`${API_URL}/analytics`);
      return response.data;
    } catch (error) {
      console.error('Analytics Error:', error);
      return null;
    }
  }
};

export default chatbotAPI;