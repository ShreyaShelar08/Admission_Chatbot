🎓 Admission Chatbot
📌 Project Description

Admission Chatbot is an AI-powered web application developed to assist students with admission-related queries.
The chatbot provides information about courses, eligibility, fees, required documents, and important dates.

The system uses DistilBERT from Hugging Face Transformers for Natural Language Processing and is deployed using a Flask backend with a React frontend.

🚀 Features

💬 Interactive chatbot interface

🤖 NLP-based response processing using DistilBERT

📚 Course and admission information support

⚡ Fast and lightweight model (DistilBERT)

🌐 REST API using Flask

🎨 Clean React-based frontend

🛠️ Technologies Used

Frontend: React.js

Backend: Flask (Python)

NLP Model: Hugging Face Transformers – DistilBERT

Libraries: transformers, torch, flask

Programming Languages: Python, JavaScript

🧠 About DistilBERT

DistilBERT is a lightweight version of BERT developed to reduce model size and increase speed while maintaining high accuracy.
It is efficient and suitable for real-time chatbot applications.

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/admission-chatbot.git
2️⃣ Backend Setup
cd backend
pip install -r requirements.txt
python app.py
3️⃣ Frontend Setup
cd frontend
npm install
npm start
📦 Required Python Libraries
pip install flask transformers torch
🔄 How It Works

User enters a query in the React interface.

The request is sent to the Flask backend via API.

DistilBERT processes the input text.

The backend generates a response.

The response is displayed in the chatbot UI.

🎯 Future Improvements

Add multilingual support

Add voice input/output

Deploy using cloud services (AWS / Render / Railway)

Improve response accuracy with fine-tuning
