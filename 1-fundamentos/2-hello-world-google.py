from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.5)
response = model.invoke("Hello, world!")

print(response.content)