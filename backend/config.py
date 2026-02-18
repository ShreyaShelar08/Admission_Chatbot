"""
Ultra-Short Configuration - Chatbot-style Responses
50-100 words max - Perfect for quick conversations!
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

# ULTRA-SHORT RESPONSES (50-100 words each)
RESPONSES = {
    "greeting": """Hi there! 👋 I'm your admission assistant.

I can help with:
• Admission process
• Fees & eligibility
• Courses & facilities
• Scholarships & placements

What would you like to know?""",

    "admission": """**Admission Process:**

1. Apply online at www.college.edu.in (₹1,000 fee)
2. Upload documents
3. Take entrance exam (June 15)
4. Attend counseling
5. Pay fees & enroll

**Deadline:** May 31, 2025

Need help? 📧 admissions@college.edu.in""",

    "fees": """**Annual Fees:**

• B.Tech CSE: ₹1,50,000
• B.Tech Others: ₹1,20,000
• MBA: ₹2,50,000
• BBA/BCA: ₹80,000
• B.Com/B.Sc: ₹50,000-60,000

**Hostel:** ₹60,000-80,000 (optional)

Payment in installments available.

📧 accounts@college.edu.in""",

    "eligibility": """**Basic Eligibility:**

**B.Tech:** 10+2 with PCM, 50%+, JEE score
**BBA/BCA:** 10+2 any stream, 50%+
**MBA:** Bachelor's degree, 50%+, CAT/MAT score
**M.Tech:** B.Tech, 55%+, GATE score

Share your qualifications for specific eligibility check!

📧 admissions@college.edu.in""",

    "deadline": """**Important Dates:**

• Applications open: Jan 1, 2025
• Last date: **May 31, 2025**
• Entrance exam: **June 15, 2025**
• Counseling: June 28 - July 10
• Classes start: **Aug 1, 2025**

⏰ Apply early!

📧 admissions@college.edu.in""",

    "contact": """**Contact Us:**

📞 Phone: +91-XXXX-XXXXXX
📧 Email: admissions@college.edu.in
💬 WhatsApp: +91-XXXXX-XXXXX

🕐 Office: Mon-Fri, 9 AM - 5 PM

🌐 Website: www.college.edu.in

📍 Address: [City, State]""",

    "facilities": """**Campus Facilities:**

📚 Library (50,000+ books)
🔬 40+ Modern Labs
🏠 Hostels (2000 capacity)
🍽️ Cafeteria & Mess
🏃 Sports Complex
🏥 Medical Center
🚌 Transport (40+ buses)
📡 Campus-wide WiFi

Want details on any specific facility?""",

    "courses": """**Programs Offered:**

**UG:** B.Tech (CSE, Mech, Civil, ECE), BBA, BCA, B.Com, B.Sc
**PG:** M.Tech, MBA, MCA, M.Sc
**Law:** BA LLB, BBA LLB (5 years)

**Total seats:** 3000+

Which course interests you?

📧 admissions@college.edu.in""",

    "scholarship": """**Scholarships Available:**

🏆 Merit: Up to 100% fee waiver (90%+ marks)
💰 Need-based: Up to 70% (income < ₹3L)
⚽ Sports: Up to 75% (state/national level)
👧 Girl child: 10% discount
🎓 Category-based: SC/ST/OBC schemes

Apply during admission!

📧 scholarship@college.edu.in""",

    "placement": """**Placement Highlights:**

✅ 98.5% placement rate
💰 Highest: ₹52 LPA
📊 Average: ₹8.5 LPA
🏢 350+ companies

**Top recruiters:** Google, Microsoft, Amazon, TCS, Infosys

Training & internships provided!

📧 placements@college.edu.in""",

    "hostel": """**Hostel Facilities:**

🏠 Separate boys & girls hostels
🛏️ Single/Double/Triple rooms
💰 ₹45,000 - ₹1,00,000/year
🍽️ Mess: ₹40,000/year (4 meals daily)
📡 WiFi, Security, Warden

Application during admission.

📧 hostel@college.edu.in""",

    "documents": """**Required Documents:**

📄 10th & 12th marksheets
📄 TC & Migration certificate
🆔 Aadhaar card
📸 10 passport photos
📝 Caste/Income certificate (if applicable)

Full list: www.college.edu.in/documents

📧 admissions@college.edu.in""",

    "exam": """**Entrance Exam:**

📅 Date: June 15, 2025
⏰ Duration: 2 hours
📝 120 MCQs (Aptitude, Reasoning, English)
💰 Fee: ₹1,000

Free mock tests available online!

📧 entranceexam@college.edu.in
☎️ 1800-XXX-XXXX""",

    "goodbye": """Thank you! 👋

Feel free to return anytime for more help!

📞 +91-XXXX-XXXXXX
📧 admissions@college.edu.in

Good luck with your admission! 🎓""",
}

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "chatbot.log"

# UI Configuration
CHAT_TITLE = "College Admission Inquiry Chatbot"
CHAT_SUBTITLE = "Get instant answers to your admission queries"
THEME_COLOR = "#667eea"
SECONDARY_COLOR = "#764ba2"