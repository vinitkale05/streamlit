from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import os


load_dotenv(dotenv_path='/Users/Vinit.Kale/Desktop/llm/.env')

model = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-1.5-flash"
)

st.header('research tool')

user_input = st.text_input('Enter your prompt ')

if st.button('summarize'):
    result = model.invoke(user_input)
    st.write(result.content if hasattr(result, "content") else result)
