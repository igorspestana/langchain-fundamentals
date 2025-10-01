from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(model="gpt-4o-mini", temperature=0.5, model_provider="openai")
response = model.invoke("Hello, world!")

print(response.content)