from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

system_prompt = ("system", "You are a helpful assistant that can answer questions in {language}.")
user_prompt = ("user", "{question}")

chat_prompt_template = ChatPromptTemplate([system_prompt, user_prompt])

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

chain = chat_prompt_template | model

response = chain.invoke({"language": "Portuguese", "question": "What is the capital of Brazil?"})

print(response.content)
