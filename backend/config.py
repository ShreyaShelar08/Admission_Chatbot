"""
Configuration file for Admission Inquiry Chatbot
"""

import os

# Model Configuration
MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = "./chatbot_model"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["*"]

# Response Configuration
RESPONSES = {
    "admission": """
    🎓 **Admission Process**
    
    Our admission process is simple and straightforward:
    
    1️⃣ **Online Application**: Visit our website and fill out the online application form
    2️⃣ **Document Submission**: Upload required documents (marksheets, ID proof, photos)
    3️⃣ **Entrance Test**: Appear for the entrance examination (if applicable for your program)
    4️⃣ **Interview**: Selected candidates will be called for a personal interview
    5️⃣ **Admission Confirmation**: Pay the admission fee to confirm your seat
    
    For more details, visit our admission portal or contact our office.
    """,
    
    "fees": """
    💰 **Fee Structure**
    
    Our fee structure varies by program:
    
    • **Undergraduate Programs**: ₹50,000 - ₹1,50,000 per year
    • **Postgraduate Programs**: ₹75,000 - ₹2,00,000 per year
    • **Professional Courses**: ₹1,00,000 - ₹3,00,000 per year
    
    📋 Additional fees may include:
    - Library fees
    - Laboratory fees
    - Sports and cultural activities
    - Hostel fees (if applicable)
    
    💳 Payment options: Semester-wise or yearly installments available.
    
    For exact fees of your specific program, please contact our accounts department or visit the fee section on our website.
    """,
    
    "eligibility": """
    ✅ **Eligibility Criteria**
    
    **For Undergraduate Programs:**
    • Completed 10+2 or equivalent from a recognized board
    • Minimum 50% aggregate marks (45% for reserved categories)
    • Age limit: 17-25 years
    
    **For Postgraduate Programs:**
    • Bachelor's degree in relevant field from a recognized university
    • Minimum 55% aggregate marks (50% for reserved categories)
    • Valid entrance test scores (if applicable)
    
    **Additional Requirements:**
    • Entrance examination (program-specific)
    • English proficiency (for international students)
    
    Note: Eligibility criteria may vary by program. Please check the specific requirements for your desired course.
    """,
    
    "deadline": """
    ⏰ **Application Deadlines**
    
    **For Academic Year 2025-26:**
    
    🗓️ **First Round:**
    - Application Start: March 1, 2025
    - Application Deadline: May 31, 2025
    - Entrance Test: June 15, 2025
    - Result Declaration: June 30, 2025
    
    🗓️ **Second Round (if seats available):**
    - Application Period: July 1 - July 31, 2025
    - Entrance Test: August 10, 2025
    - Result Declaration: August 20, 2025
    
    ⚠️ **Important Notes:**
    - Late applications may be accepted with a late fee
    - International students should apply at least 3 months in advance
    - Spot admissions may be available for certain programs
    
    Don't miss the deadline! Apply early to ensure your seat.
    """,
    
    "contact": """
    📞 **Contact Information**
    
    **Admission Office:**
    • 📧 Email: admissions@college.edu.in
    • 📱 Phone: +91-XXXX-XXXXXX
    • 📠 Fax: +91-XXXX-XXXXXX
    
    **Office Address:**
    [College Name]
    [Address Line 1]
    [Address Line 2]
    [City, State - PIN Code]
    
    **Office Hours:**
    • Monday - Friday: 9:00 AM - 5:00 PM
    • Saturday: 9:00 AM - 1:00 PM
    • Sunday: Closed
    
    **Social Media:**
    • 🌐 Website: www.college.edu.in
    • 📘 Facebook: /collegename
    • 📸 Instagram: @collegename
    • 🐦 Twitter: @collegename
    
    **Emergency Contact:** +91-XXXX-XXXXXX (24/7)
    
    Feel free to reach out to us for any queries!
    """,
    
    "greeting": """
    👋 Hello! Welcome to our **College Admission Inquiry Chatbot**.
    
    I'm here to help you with:
    • Admission process and procedures
    • Fee structure and payment details
    • Eligibility criteria
    • Application deadlines
    • Contact information
    • And much more!
    
    Feel free to ask me anything about admissions. How can I assist you today?
    """,
    
    "goodbye": """
    👋 Thank you for using our admission inquiry service!
    
    We hope we were able to help you with your queries. 
    
    If you have any more questions in the future, feel free to come back anytime. 
    
    **Good luck with your admission!** 🎓✨
    
    Have a great day! 😊
    """,
}

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "chatbot.log"

# Database Configuration (for future use)
DB_PATH = "./chatbot_data.db"

# UI Configuration
CHAT_TITLE = "College Admission Inquiry Chatbot"
CHAT_SUBTITLE = "Get instant answers to your admission queries"
THEME_COLOR = "#667eea"
SECONDARY_COLOR = "#764ba2"