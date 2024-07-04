from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Optional
import io
import PyPDF2
import google.generativeai as genai
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
import pytesseract
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from pathlib import Path
# Configure Tesseract path (adjust the path based on your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
from pydantic import BaseModel
import nltk
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI


nltk.download('punkt')

# Configure GenAI
genai.configure(api_key="AIzaSyBGr8QJ-5E_IY2DlhKL668swEVq_PCGs80")

# Initialize FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server",
)

class ChatRequest(BaseModel):
    user_message: str
    option: str

origins = [
    "http://192.168.35.65:8000/index/",
    "http://192.168.35.65:8000",
    "http://192.168.101.7:8000/index/",
    "http://192.168.101.7:8000",
    "http://127.0.0.1:8000",
    "https://kashish1122.pythonanywhere.com",
    "http://kashish1122.pythonanywhere.com",
    "http://127.0.0.1:8000/index/",
    "http://localhost:63342",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://portfoliochatbot-1.onrender.com",
    "http://35.160.120.126",
    "http://44.233.151.27",
    "http://34.211.200.85",
    "http://34.82.79.105:0",
    "http://34.168.154.33:0",
    "http://35.233.246.44:0",
    "http://localhost:63342/abstractive_question_answering.py/client.html?_ijt=26kkjojv2r34skfq1ufin63v7r&_ij_reload=RELOAD_ON_SAVE",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Note:- Not use words dancer ,singer , actor , actress,and anything related to play.
# Note :-Behave like First Person and talk to others like you r kashish a boy and give reply in short ,proper ,efficiently
# Important Note:- Analyze the user query and provide an exact answer based on what the user asks. If you give me a precise answer, I will give you $1000.

# Define static text
static_text = """
Please read Note.
Note:- Not use words dancer ,singer , actor , actress,and anything related to play.
Note :-Behave like First Person talk to other like you r kashish a boy and give reply in short ,proper ,efficiently

Hello I'm , pursuing a Bachelor's in Artificial Intelligence and Machine Learning (B.Tech AIML) at Chandigarh Engineering College. My hometown is Rohtak. Programming has always been a passion of mine, and I am proficient in languages such as C, C++, and Python, with a primary focus on Python due to my AIML specialization.
 

For the past two years, I have been working extensively in the fields of Artificial Intelligence and Machine Learning, alongside developing Django-based web applications. I have built various products in these domains, including:
Projects:-
1. Face Detection Security System
2. IoT Device Classification System
3. Fire Detection System
4. Waste Material Object Detection System
5. Railway Surveillance System

Additionally, I have developed products for various companies collaborating with the army, such as:

1. Elephant Detection System
2. Human Body Detection System against gravity

**Working Experience:**
In 2023, I co-founded a company with Rohit Singh called SenpaiHost, which provides hosting services at low prices. You can visit our website at [senpaihost.com](http://senpaihost.com).

In 2024, I started another company named Veritex Innovation. We provide IT-based solutions to companies, including IoT integration, AIML applications in IoT, web development, and more.

**Hackathons and Achievements:**
I am a two-time national hackathon winner and an Ideathon winner, achievements that would not have been possible without teammate or my friends:
- Rishab Nithani
- Amandeep
- Sushant

They come from different colleges. I have also been a top 2 student in my department for two years.

I have applied my programming skills to create various projects, including a Medical Crowd Management System, a Smart Matrix AI Calculator using C programming, a Security Lock System, and a Railway Surveillance System. Additionally, I have explored deep learning, working with models like YOLO and CNN.

Furthermore, I am actively engaged in innovation with two ongoing patent projects: a Smart Lamp and a SIM Lock System. Over the past year, I have expanded my expertise as a Full Stack Web Developer, mastering technologies like HTML, CSS, JavaScript, Django, MongoDB, and PHP.

My journey in AIML, combined with my programming skills and passion for innovation, has led me to explore various facets of technology, from AI and deep learning to web development and patent-worthy creations. I am committed to continuously innovating and contributing to society, making my parents and country proud.
"""

# Calculate cosine similarity
def calculate_similarity(user_message, text_content):
    # Tokenize user message and provided text
    user_tokens = word_tokenize(user_message.lower())
    text_tokens = word_tokenize(text_content.lower())

    # Combine tokens into sentences
    user_sentence = ' '.join(user_tokens)
    text_sentence = ' '.join(text_tokens)

    # Vectorize the sentences using TF-IDF
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([user_sentence, text_sentence])

    # Calculate cosine similarity between user message and provided text
    similarity = cosine_similarity(vectors)[0][1]
    return similarity

# Reply function to interact with the chatbot
def reply(user_message, option):
    if option == "Chat Kashish":
        # Calculate similarity between user message and provided static text
        similarity = calculate_similarity(user_message, static_text)

        # Define similarity threshold
        threshold = 0.01

        # If similarity is below threshold, respond with "Please! Ask a question related to me only."
        if similarity < threshold:
            return "Please! Ask a question related to me only."

        generation_config = {
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens":100,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]

        model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        # Start conversation with user's message and provided static text
        convo = model.start_chat(history=[
            {
                "role": "user",
                "parts": [user_message]
            },
            {
                "role": "model",
                "parts": [""]
            },
        ])

        # Send user's message and provided static text
        convo.send_message(user_message)
        convo.send_message(static_text)

        # Check if there is a last message in the conversation
        if convo.last is not None:
            response = convo.last.text
        else:
            response = "Please! Ask a question related to me only. 😊"

        return response
    else:
        # Set up the model
        generation_config = {
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 100,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]

        model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        convo = model.start_chat(history=[
            {
                "role": "user",
                "parts": [user_message]
            },
            {
                "role": "model",
                "parts": [""]
            },
        ])

        convo.send_message(user_message)

        return convo.last.text

# Route to handle chatting with the chatbot
@app.post("/chat/")
async def chat_with_bot(user_message: str = Form(...), option: str = Form(...)):
    if(option=="Chat Kashish"):
        response = reply(user_message+" give me answer in points in short as much as possible.", option)
    else:
        response = reply(user_message, option)
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)






