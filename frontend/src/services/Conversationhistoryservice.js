// src/services/conversationHistoryService.js
// Conversation-based history (like ChatGPT) - NOT day-based

export const conversationHistoryService = {
  // Generate unique conversation ID
  generateConversationId: () => {
    return `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  },

  // Save entire conversation
  saveConversation: (userEmail, conversationId, messages) => {
    try {
      if (!messages || messages.length === 0) return false;

      const conversation = {
        id: conversationId,
        messages: messages,
        title: conversationHistoryService.generateTitle(messages),
        createdAt: messages[0].timestamp,
        updatedAt: messages[messages.length - 1].timestamp,
        messageCount: messages.length
      };

      // Get all user conversations
      const allConversations = conversationHistoryService.getAllConversations(userEmail);
      
      // Update or add this conversation
      const existingIndex = allConversations.findIndex(c => c.id === conversationId);
      if (existingIndex !== -1) {
        allConversations[existingIndex] = conversation;
      } else {
        allConversations.unshift(conversation); // Add to beginning
      }

      // Save back
      localStorage.setItem(`conversations_${userEmail}`, JSON.stringify(allConversations));
      
      return true;
    } catch (error) {
      console.error('Error saving conversation:', error);
      return false;
    }
  },

  // Generate conversation title from first message
  generateTitle: (messages) => {
    if (!messages || messages.length === 0) return "New Chat";
    
    const firstUserMessage = messages.find(msg => msg.sender === 'user');
    if (firstUserMessage) {
      const text = firstUserMessage.text.trim();
      // Take first sentence or 50 characters
      const firstSentence = text.split(/[.!?]/)[0];
      return firstSentence.length > 50 
        ? firstSentence.substring(0, 50) + '...' 
        : firstSentence;
    }
    
    return "New Chat";
  },

  // Get all conversations for user
  getAllConversations: (userEmail) => {
    try {
      const data = localStorage.getItem(`conversations_${userEmail}`);
      return data ? JSON.parse(data) : [];
    } catch (error) {
      console.error('Error getting conversations:', error);
      return [];
    }
  },

  // Get single conversation by ID
  getConversation: (userEmail, conversationId) => {
    try {
      const allConversations = conversationHistoryService.getAllConversations(userEmail);
      return allConversations.find(c => c.id === conversationId);
    } catch (error) {
      console.error('Error getting conversation:', error);
      return null;
    }
  },

  // Delete conversation
  deleteConversation: (userEmail, conversationId) => {
    try {
      const allConversations = conversationHistoryService.getAllConversations(userEmail);
      const filtered = allConversations.filter(c => c.id !== conversationId);
      localStorage.setItem(`conversations_${userEmail}`, JSON.stringify(filtered));
      return true;
    } catch (error) {
      console.error('Error deleting conversation:', error);
      return false;
    }
  },

  // Update conversation title
  updateTitle: (userEmail, conversationId, newTitle) => {
    try {
      const allConversations = conversationHistoryService.getAllConversations(userEmail);
      const conversation = allConversations.find(c => c.id === conversationId);
      
      if (conversation) {
        conversation.title = newTitle;
        localStorage.setItem(`conversations_${userEmail}`, JSON.stringify(allConversations));
        return true;
      }
      return false;
    } catch (error) {
      console.error('Error updating title:', error);
      return false;
    }
  },

  // Get conversations organized by time periods
  getOrganizedConversations: (userEmail) => {
    try {
      const allConversations = conversationHistoryService.getAllConversations(userEmail);
      
      const organized = {
        "Today": [],
        "Yesterday": [],
        "Last 7 Days": [],
        "Last 30 Days": [],
        "Older": []
      };

      const now = new Date();
      const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
      const yesterday = new Date(today);
      yesterday.setDate(yesterday.getDate() - 1);

      allConversations.forEach(conv => {
        const convDate = new Date(conv.updatedAt);
        const convDateOnly = new Date(convDate.getFullYear(), convDate.getMonth(), convDate.getDate());
        
        const daysDiff = Math.floor((today - convDateOnly) / (1000 * 60 * 60 * 24));

        let period;
        if (convDateOnly.getTime() === today.getTime()) {
          period = "Today";
        } else if (convDateOnly.getTime() === yesterday.getTime()) {
          period = "Yesterday";
        } else if (daysDiff <= 7) {
          period = "Last 7 Days";
        } else if (daysDiff <= 30) {
          period = "Last 30 Days";
        } else {
          period = "Older";
        }

        organized[period].push({
          id: conv.id,
          title: conv.title,
          time: new Date(conv.updatedAt).toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
          }),
          messageCount: conv.messageCount,
          date: new Date(conv.updatedAt).toLocaleDateString()
        });
      });

      // Remove empty periods
      Object.keys(organized).forEach(key => {
        if (organized[key].length === 0) {
          delete organized[key];
        }
      });

      return organized;
    } catch (error) {
      console.error('Error organizing conversations:', error);
      return {};
    }
  },

  // Clear all conversations for user
  clearAllConversations: (userEmail) => {
    try {
      localStorage.removeItem(`conversations_${userEmail}`);
      return true;
    } catch (error) {
      console.error('Error clearing conversations:', error);
      return false;
    }
  },

  // Export all conversations
  exportConversations: (userEmail) => {
    try {
      const allConversations = conversationHistoryService.getAllConversations(userEmail);
      return {
        user: userEmail,
        exportedAt: new Date().toISOString(),
        totalConversations: allConversations.length,
        conversations: allConversations
      };
    } catch (error) {
      console.error('Error exporting conversations:', error);
      return null;
    }
  }
};

export default conversationHistoryService;