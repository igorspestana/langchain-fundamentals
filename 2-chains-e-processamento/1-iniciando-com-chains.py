from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

template = PromptTemplate(
    input_variables=["name"],
    template="Hello, i'm {name}!",
)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

chain = template | model

response = chain.invoke({"name": "Pedro"})

print(response.content)