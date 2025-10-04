from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

def square(x: int) -> dict:
    return {"square_result": x * x, "x": x}

square_lambda = RunnableLambda(square)

question_template = PromptTemplate(
    input_variables=["x"],
    template="What is the square of {x}? Is it {square_result}?",
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

chain = square_lambda | question_template | model

response = chain.invoke(4)

print(response.content)