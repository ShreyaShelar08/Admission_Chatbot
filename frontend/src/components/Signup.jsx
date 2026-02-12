import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Signup.css";
import botImage from "./AI-Chatbot-Banner.webp";

const Signup = () => {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleSignup = (e) => {
    e.preventDefault();

    if (!fullName || !email || !password) {
      alert("Please fill in all fields");
      return;
    }

    setIsLoading(true);

    setTimeout(() => {
      const users = JSON.parse(localStorage.getItem("users")) || [];

      if (users.find((u) => u.email === email)) {
        alert("User already exists with this email");
        setIsLoading(false);
        return;
      }

      const newUser = { 
        fullName, 
        email, 
        password,
        createdAt: new Date().toISOString()
      };
      
      users.push(newUser);
      localStorage.setItem("users", JSON.stringify(users));

      const submitBtn = e.target.querySelector('.signup-btn');
      if (submitBtn) {
        submitBtn.style.background = 'linear-gradient(135deg, #10b981 0%, #34d399 100%)';
        submitBtn.innerHTML = '✓ Signup Successful';
      }

      setTimeout(() => {
        alert("Signup successful! Please login with your credentials.");
        navigate("/");
      }, 1000);
    }, 1000);
  };

  const handleSocialLogin = (provider) => {
    alert(`${provider} authentication would require OAuth setup with backend.\n\nFor now, please use email signup.\n\nTo enable ${provider} login:\n1. Create app in ${provider} Developer Console\n2. Get API keys\n3. Implement OAuth in backend\n4. Connect to React app`);
  };

  return (
    <div className="signup-wrapper">
      {/* LEFT SECTION */}
      <div className="signup-left">
        <div className="logo">DUX AI</div>

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

      {/* RIGHT SECTION - Compact layout */}
      <div className="signup-right">
        {/* "Already have account" link */}
        <div className="top-login">
          Already have an account?{" "}
          <span onClick={() => navigate("/")}>Sign In</span>
        </div>

        {/* Main content container */}
        <div className="signup-main-content">
          <h2>Create Account</h2>

          <form onSubmit={handleSignup} className="signup-form">
            <div className="form-group">
              <label>Full Name</label>
              <input
                type="text"
                placeholder="John Doe"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                required
                autoFocus
              />
            </div>

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
                placeholder="Create a secure password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength="6"
              />
            </div>

            <div className="checkbox-container">
              <input type="checkbox" id="terms" required />
              <label htmlFor="terms">
                I agree to the{" "}
                <a href="#" onClick={(e) => { e.preventDefault(); alert("Terms of Service page would go here"); }}>
                  Terms of Service
                </a>{" "}
                and{" "}
                <a href="#" onClick={(e) => { e.preventDefault(); alert("Privacy Policy page would go here"); }}>
                  Privacy Policy
                </a>.
              </label>
            </div>

            <button 
              type="submit" 
              className="signup-btn"
              disabled={isLoading}
            >
              {isLoading ? "Creating Account..." : "Sign Up"}
            </button>
          </form>

          {/* Social Login Section */}
          <div className="social-login-section">
            <div className="divider">Or Continue With</div>

            <div className="social-buttons">
              <button 
                className="social-btn google" 
                onClick={() => handleSocialLogin("Google")}
                type="button"
              >
                Google
              </button>
              <button 
                className="social-btn github" 
                onClick={() => handleSocialLogin("GitHub")}
                type="button"
              >
                GitHub
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Signup;