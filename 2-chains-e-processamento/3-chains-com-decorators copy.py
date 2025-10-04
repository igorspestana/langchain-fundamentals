from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from dotenv import load_dotenv

load_dotenv()


# @chain
# def square(input_dict: dict) -> dict:
#     x = input_dict["x"]
#     return {"square_result": x * x, "x": x}

@chain
def square(x: int) -> dict:
    return {"square_result": x * x, "x": x}


question_template = PromptTemplate(
    input_variables=["x"],
    template="What is the square of {x}? Is it {square_result}?",
)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

chain = square | question_template | model

# response = chain.invoke({"x": 2})
response = chain.invoke(4)

print(response.content)