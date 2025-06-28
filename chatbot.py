from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='/Users/Vinit.Kale/Desktop/llm/.env')

model = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-1.5-flash"
)

chat_history = [
    SystemMessage(content='you are helpfull assistant')
]


while True:
    user_input = input("you: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chatbot. goodbye!")
        break
    result = model.invoke(user_input)
    chat_history.append(AIMessage(content=result.content if hasattr(result, "content") else result))
    print("ai :", result.content if hasattr(result, "content") else result)