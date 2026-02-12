import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "./Login.css";
import botImage from "./AI-Chatbot-Banner.webp"; // 👈 add image in same folder or adjust path

const Login = ({ setUser }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const currentUser = localStorage.getItem("currentUser");
    if (currentUser) {
      navigate("/dashboard");
    }
  }, [navigate]);

  const handleLogin = (e) => {
    e.preventDefault();

    if (!email || !password) {
      alert("Please fill in all fields");
      return;
    }

    setIsLoading(true);

    setTimeout(() => {
      const users = JSON.parse(localStorage.getItem("users")) || [];
      const existingUser = users.find(
        (u) => u.email === email && u.password === password
      );

      if (existingUser) {
        setUser(existingUser);
        localStorage.setItem("currentUser", JSON.stringify(existingUser));

        setTimeout(() => navigate("/dashboard"), 1000);
      } else {
        alert("Invalid credentials. Please try again.");
        setIsLoading(false);
      }
    }, 1000);
  };

  return (
    <div className="login-wrapper">
      {/* LEFT SECTION */}
      <div className="login-left">
        <div className="logo">DUX AI</div>

        {/* 🔥 AI BOT IMAGE (REPLACED CENTER OBJECT) */}
        <div className="bot-image-wrapper">
          <img src={botImage} alt="AI Bot" />
        </div>

        <h1>
          Welcome to <span>DUX AI</span>
        </h1>
        <p>
          An AI-based chatbot developed to enable intelligent, human-like interaction between users and systems.
        </p>
      </div>

      {/* RIGHT SECTION */}
      <div className="login-right">
        <div className="top-signup">
          New to DUX?
          <span onClick={() => navigate("/signup")}>Create Account</span>
        </div>

        <h2>Sign In</h2>
        <p className="login-subtitle">
          Enter your credentials to access the chatbot
        </p>

        <form onSubmit={handleLogin} className="login-form">
          <div className="form-group">
            <label>Email Address</label>
            <input
              type="email"
              placeholder="name@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <div className="forgot-password">
            <a href="#">Forgot Password?</a>
          </div>

          <button type="submit" className="login-btn" disabled={isLoading}>
            {isLoading ? "Signing In..." : "Sign In"}
          </button>
        </form>
      </div>
    </div>
  );
};

export default Login;
