from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

system_prompt = ("system", "You are a helpful assistant that can answer questions in {language}.")
user_prompt = ("user", "{question}")

chat_prompt_template = ChatPromptTemplate([system_prompt, user_prompt])

messages = chat_prompt_template.format_messages(language= "English", question= "What is the capital of France?")

for message in messages:
    print(f"{message.type}: {message.content}")

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
response = model.invoke(messages)

print(response.content)