from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["name"],
    template="Hello, {name}!",
)

print(template.format(name="John"))

print(template.invoke({"name": "John"}))

print(template.invoke({"name": "Jane"}))