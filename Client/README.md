# 🌱 GreenGenie – Renewable Energy Awareness Chatbot


An AI-powered chatbot designed to spread awareness about renewable energy sources and promote sustainable practices. Built to educate users with accurate, real-time, and conversational responses using Natural Language Processing (NLP) and vector-based semantic search.

## 🚀 Project Overview

This chatbot aims to:
- Provide instant information on various renewable energy sources.
- Answer questions about solar, wind, hydro, biomass, and geothermal energy.
- Educate users on environmental impact and sustainable practices.
- Support the **United Nations Sustainable Development Goal 7 (Affordable and Clean Energy)**.

## 🧠 Features

- 💬 Interactive Chat Interface (Mobile & Web)
- 🔍 Semantic Search using FAISS & Vector Database
- 📄 Knowledge base with chunked data for accurate answers
- 🤖 Powered by AI (OpenAI/Gemini for LLM responses)
- 🌐 Awareness content from trusted sources like MNRE, UN, and other energy bodies
- 📈 Scalable backend with Express.js and Flask


### 💻 Project Structure – GreenGeenie Chatbot

| Layer                 | Responsibility                         | Technology Used                                   |
| --------------------- | -------------------------------------- | ------------------------------------------------- |
| *Frontend*          | Basic user interface to ask questions  | HTML + JavaScript (or React if extended)          |
| *Backend*           | REST API, RAG pipeline, response logic | *Python, **Flask, **LangChain, **FAISS*   |
| *AI Model*          | Natural language understanding         | *Gemini API* (via langchain_google_genai)     |
| *Document Handling* | Text extraction, chunking              | PyMuPDF / Custom Loader + LangChain Chunking      |
| *Database*          | Vector storage for document search     | *FAISS*                                         |
| *Prompting*         | Response control & fallback logic      | PromptTemplate (LangChain) + Custom Conditions    |
| *Deployment*        | App hosting                            | *Render* (Backend), Optional Netlify (Frontend) |
| *Version Control*   | Code collaboration                     | Git & GitHub                                      |



### 👥 Team Contributions (JOJO Coders)

| Member               | Role                              |
| -------------------- | --------------------------------- |
| Vijaya Vardhan Killi | Backend (Flask, LangChain, FAISS) |
| Md Sharieff          | Frontend (React, UI, integration) |
| Davud Shaik          | Testing & research support        |
| Rajesh Mummidi       | Documentation & content review    |


## 🧩 How It Works

1. **User Interaction** – A user asks a question via the chatbot.
2. **Text Processing** – The query is converted to a vector using an embedding model.
3. **Semantic Search** – FAISS retrieves relevant chunks from the knowledge base.
4. **AI Response** – A language model (LLM) processes the context and generates a user-friendly answer.
5. **Response Delivery** – The chatbot displays the answer in real time.


## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/MdSharieff-081/renewable-energy-chatbot.git
cd renewable-energy-chatbot

# Backend setup
cd backend
npm install
npm start

# Frontend setup
cd ../frontend
npm install
npm start

```

## 📚 Data Sources

MNRE (Ministry of New and Renewable Energy, India)

UN SDG resources

Academic & government research papers

Clean energy blogs and PDFs (converted into chunks)



## 💡 Future Enhancements

📲 Add voice support for accessibility

🔒 Student/Teacher login system

🧾 Weekly quiz or awareness challenges

📊 Dashboard for tracking awareness impact

🌐 Multi-language support



## 🧑‍💻 Contributing

Contributions are welcome! Please open issues and submit pull requests for improvements, bug fixes, or content updates.


## 📜 License

This project is licensed under the MIT License.


## 🙌 Acknowledgements

OpenAI

Gemini

FAISS by Facebook AI

LangChain

MNRE


## 🌟 Final Note

The Renewable Energy Awareness Chatbot is more than just a tech project — it's a step toward building a sustainable future through education and empowerment. By making clean energy knowledge accessible and engaging, we hope to inspire action and responsibility toward our planet.

🔋 Let's harness the power of AI for a greener tomorrow.  
🤝 Together, we can drive awareness, one conversation at a time.








