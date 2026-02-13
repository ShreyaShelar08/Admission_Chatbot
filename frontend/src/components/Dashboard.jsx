import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import chatbotAPI from "../services/api";
import conversationHistoryService from '../services/Conversationhistoryservice';
import "./Dashboard.css";

const Dashboard = ({ user, setUser }) => {
  // ========== STATE DECLARATIONS ==========
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [chatHistory, setChatHistory] = useState({});
  const [showWelcome, setShowWelcome] = useState(true);
  
  const messagesEndRef = useRef(null);
  const navigate = useNavigate();

  // ========== HELPER FUNCTIONS ==========

  const getTimePeriod = (dateString) => {
    const date = new Date(dateString);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    
    const dateOnly = dateString.split('T')[0];
    const todayOnly = today.toISOString().split('T')[0];
    const yesterdayOnly = yesterday.toISOString().split('T')[0];
    
    if (dateOnly === todayOnly) return "Today";
    if (dateOnly === yesterdayOnly) return "Yesterday";
    
    const daysDiff = Math.floor((today - date) / (1000 * 60 * 60 * 24));
    if (daysDiff <= 7) return "Last 7 Days";
    if (daysDiff <= 30) return "Last 30 Days";
    
    return "Older";
  };

  const generateConversationTitle = (messages) => {
    if (!messages || messages.length === 0) return "New Chat";
    
    const firstUserMessage = messages.find(msg => msg.sender === 'user');
    if (firstUserMessage) {
      const text = firstUserMessage.text;
      return text.length > 40 ? text.substring(0, 40) + '...' : text;
    }
    
    return "New Chat";
  };

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // ========== CONVERSATION MANAGEMENT ==========

  const loadChatHistory = () => {
    if (!user?.email) return;
    const organized = conversationHistoryService.getOrganizedConversations(user.email);
    setChatHistory(organized);
  };

  const startNewChat = () => {
    // Save current conversation before starting new
    if (messages.length > 0 && currentConversationId && user?.email) {
      conversationHistoryService.saveConversation(
        user.email,
        currentConversationId,
        messages
      );
      loadChatHistory();
    }
   
    // Create new conversation
    const newConvId = conversationHistoryService.generateConversationId();
    setCurrentConversationId(newConvId);
    
  };

  const handleHistoryClick = (conversationId) => {
    // Save current first
    if (messages.length > 0 && currentConversationId && user?.email) {
      conversationHistoryService.saveConversation(
        user.email,
        currentConversationId,
        messages
      );
    }

    // Load selected conversation
    const conversation = conversationHistoryService.getConversation(user.email, conversationId);
    if (conversation) {
      setMessages(conversation.messages);
      setCurrentConversationId(conversationId);
      setShowWelcome(false);
    }
  };

  const handleDeleteConversation = (conversationId, e) => {
    e.stopPropagation();
    
    if (window.confirm('Delete this conversation?')) {
      conversationHistoryService.deleteConversation(user.email, conversationId);
      
      // If deleting current conversation, start new one
      if (conversationId === currentConversationId) {
        startNewChat();
      }
      
      loadChatHistory();
    }
  };

  // ========== MESSAGE HANDLING ==========

  const handleSend = async (e, customMessage = null) => {
    if (e && e.preventDefault) e.preventDefault();
    
    const messageText = customMessage || input;
    if (!messageText.trim()) return;

    // Hide welcome message on first user message
    if (showWelcome) {
      setShowWelcome(false);
    }

    const userMessage = {
      id: Date.now(),
      text: messageText,
      sender: "user",
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      timestamp: new Date().getTime()
    };

    setMessages(prev => [...prev, userMessage]);
    
    if (!customMessage) setInput("");
    setIsTyping(true);

    try {
      const result = await chatbotAPI.sendMessage(messageText);
      
      if (result.success) {
        const botResponse = {
          id: Date.now() + 1,
          text: result.data.response,
          sender: "bot",
          intent: result.data.intent,
          confidence: result.data.confidence,
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          timestamp: new Date().getTime()
        };

        setMessages(prev => [...prev, botResponse]);
        
      } else {
        const errorResponse = {
          id: Date.now() + 1,
          text: "⚠️ Backend connection failed. Please make sure Python backend is running.",
          sender: "bot",
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          timestamp: new Date().getTime()
        };
        setMessages(prev => [...prev, errorResponse]);
      }
      
    } catch (error) {
      console.error('Error sending message:', error);
      const errorResponse = {
        id: Date.now() + 1,
        text: "❌ Connection error. Make sure the Python backend is running on port 8000.",
        sender: "bot",
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        timestamp: new Date().getTime()
      };
      setMessages(prev => [...prev, errorResponse]);
    } finally {
      setIsTyping(false);
    }
  };

  // ========== WELCOME & QUICK ACTIONS ==========

  const handleWelcomeTopicClick = (topic, question) => {
    setShowWelcome(false);
    handleSend(null, question);
  };

  const handleQuickQuestion = (questionText) => {
    if (showWelcome) {
      setShowWelcome(false);
    }
    handleSend(null, questionText);
  };

  // ========== OTHER HANDLERS ==========

  const handleLogout = () => {
    // Save current conversation before logout
    if (messages.length > 0 && currentConversationId && user?.email) {
      conversationHistoryService.saveConversation(
        user.email,
        currentConversationId,
        messages
      );
    }
    
    localStorage.removeItem("currentUser");
    if (setUser) {
      setUser(null);
    }
    navigate("/");
  };

  const handleExportChat = () => {
    const chatData = messages.map(msg => ({
      sender: msg.sender === "bot" ? "DUX" : "You",
      time: msg.time,
      text: msg.text
    }));
    
    const jsonData = JSON.stringify(chatData, null, 2);
    const blob = new Blob([jsonData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    alert("Chat exported as JSON file!");
  };

  const handleExportHistory = () => {
    const allHistory = conversationHistoryService.exportConversations(user.email);
    const blob = new Blob([JSON.stringify(allHistory, null, 2)], 
      { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat_history_${user.email}_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleNewChat = () => {
    setMessages([
      {
        id: 1,
        text: "Hello! I'm your AI assistant. What would you like to discuss today?",
        sender: "bot",
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
  };

  // ========== EFFECTS ==========

  // Check authentication
  useEffect(() => {
    const currentUser = localStorage.getItem("currentUser");
    if (!currentUser && !user) {
      navigate("/");
    }
  }, [navigate, user]);

  // Initialize conversation on mount
  useEffect(() => {
    if (user?.email) {
      loadChatHistory();
      startNewChat();
    }
  }, [user]);

  // Auto-save conversation when messages change
  useEffect(() => {
    if (messages.length > 0 && currentConversationId && user?.email) {
      conversationHistoryService.saveConversation(
        user.email,
        currentConversationId,
        messages
      );
      loadChatHistory();
    }
  }, [messages]);

  // Auto-scroll
  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  // ========== RENDER ==========
  
  return (
    <div className="dashboard-container">
      {/* LEFT SIDEBAR */}
      <div className="sidebar-left">
        {/* Brand / App Identity */}
        <div className="sidebar-header">
          <div className="logo-section">
            <div className="logo-icon">⚡</div>
            <h1 className="app-title">DUX</h1>
          </div>
        </div>

        {/* User Profile (Compact) */}
        <div className="user-profile-section">
          <div className="user-avatar">
            {user?.fullName?.charAt(0) || user?.email?.charAt(0) || "Y"}
          </div>
          <div className="user-name">{user?.fullName || "yugandhara patil"}</div>
          <button className="logout-btn" onClick={handleLogout}>
            Logout
          </button>
        </div>

        {/* Chat History Section */}
        <div className="chat-history-section">
          {Object.entries(chatHistory).map(([period, items]) => (
            <div key={period} className="history-group">
              <h4 className="history-period">{period}</h4>
              <div className="history-items">
                {items.map((item) => (
                  <div
                    key={item.id}
                    className="history-item"
                    onClick={() => handleHistoryClick(item.id)}
                  >
                    <div className="history-content">
                      <div className="history-title">{item.title}</div>
                      <div className="history-time">{item.time}</div>
                    </div>
                    <button
                      className="delete-btn"
                      onClick={(e) => handleDeleteConversation(item.id, e)}
                      title="Delete conversation">
                      🗑️
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* MAIN CONTENT AREA */}
      <div className="main-content">
        {/* Top Header Section - Centered */}
        <div className="main-header">
          <div className="greeting-section">
            <h2 className="greeting-text">Hello, {user?.fullName?.split(' ')[0] || "Yugandhara"}!</h2>
            <p className="greeting-subtext">How can I assist you today?</p>
          </div>
        </div>

        {/* Chat Area (Conversation Panel) */}
        <div className="chat-area">
          <div className="messages-container">
            {/* Welcome Message with Topics */}
            {showWelcome && (
              <div className="welcome-message">
                <h2>👋 Welcome to DUX!</h2>
                <p>I'm here to help you with all your admission-related questions.</p>
                
                <div className="welcome-topics">
                  <div 
                    className="welcome-topic" 
                    onClick={() => handleWelcomeTopicClick("admission", "What is the admission process?")}
                  >
                    <div className="welcome-topic-icon">📋</div>
                    <div className="welcome-topic-title">Admission Process</div>
                  </div>
                  <div 
                    className="welcome-topic" 
                    onClick={() => handleWelcomeTopicClick("fees", "What are the fees?")}
                  >
                    <div className="welcome-topic-icon">💰</div>
                    <div className="welcome-topic-title">Fee Structure</div>
                  </div>
                  <div 
                    className="welcome-topic" 
                    onClick={() => handleWelcomeTopicClick("eligibility", "Am I eligible?")}
                  >
                    <div className="welcome-topic-icon">✅</div>
                    <div className="welcome-topic-title">Eligibility</div>
                  </div>
                  <div 
                    className="welcome-topic" 
                    onClick={() => handleWelcomeTopicClick("deadlines", "What is the deadline?")}
                  >
                    <div className="welcome-topic-icon">⏰</div>
                    <div className="welcome-topic-title">Deadlines</div>
                  </div>
                </div>
              </div>
            )}

            {/* Chat Messages */}
            {messages.map((msg) => (
              <div 
                key={msg.id} 
                className={`message-bubble ${msg.sender}`}
              >
                <div className="message-header">
                  <div className="sender-info">
                    <span className="sender-icon">
                      {msg.sender === "bot" ? "🤖" : "👤"}
                    </span>
                    <span className="sender-name">
                      {msg.sender === "bot" ? "DUX" : "You"}
                    </span>
                  </div>
                  <span className="message-time">{msg.time}</span>
                </div>
                <div className="message-text">{msg.text}</div>
              </div>
            ))}
            
            {/* Typing Indicator */}
            {isTyping && (
              <div className="typing-indicator">
                <div className="typing-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <span className="typing-text">DUX is typing...</span>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Section (Bottom) */}
        <div className="input-section">
          <form className="input-form" onSubmit={handleSend}>
            <input
              type="text"
              className="chat-input"
              placeholder="Type your message here..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              autoFocus
            />
            <button type="submit" className="send-btn">
              <span className="send-icon">→</span>
            </button>
          </form>
        </div>
      </div>

      {/* RIGHT SIDEBAR - Quick Actions & Quick Questions */}
      <div className="sidebar-right">
        {/* Quick Actions */}
        <div className="sidebar-card">
          <h3>⚡ Quick Actions</h3>
          <div className="action-buttons-sidebar">
            <button className="action-button" onClick={handleNewChat}>
            <span className="plus-icon">+</span> New Chat
          </button>
            <button className="action-button" onClick={handleExportChat}>
              📥 Export Chat
            </button>
            <button className="action-button" onClick={handleNewChat}>
              🗑️ Clear Chat
            </button>
          </div>
        </div>

        {/* Quick Questions */}
        <div className="sidebar-card">
          <h3>💬 Quick Questions</h3>
          <div className="quick-questions">
            <div className="quick-question" onClick={() => handleQuickQuestion("Hello")}>
              👋 Say Hello
            </div>
            <div className="quick-question" onClick={() => handleQuickQuestion("How do I apply?")}>
              📝 Application Process
            </div>
            <div className="quick-question" onClick={() => handleQuickQuestion("What is the contact information?")}>
              📞 Contact Details
            </div>
            <div className="quick-question" onClick={() => handleQuickQuestion("Thank you")}>
              🙏 Say Thanks
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};


export default Dashboard;