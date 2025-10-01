from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model="gemini-2.0-flash-exp", temperature=0.5, model_provider="google_genai")
response = model.invoke("Hello, world!")

print(response.content)