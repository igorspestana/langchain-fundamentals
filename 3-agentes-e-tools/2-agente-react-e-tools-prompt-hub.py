from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv
from langchain import hub

load_dotenv()

# create a tool to evaluate a simple math expression
# return_direct=True means that the tool will return the result directly to the user instead of returning the result to the agent
@tool("calculator", return_direct=True)
def calculator(question: str) -> str:
    """Use this tool to evaluate a simple math expression and return the result"""
    try:
        result = eval(question) # be careful with this, it's a security risk
    except Exception as e:
        return f"Error: {e}"
    return str(result)

@tool("web_search_mock")
def web_search_mock(query: str) -> str:
    """Use this tool to search the mock data for information"""

    # mock data for the web search
    data = {
        "Brazil": "Brasília", 
        "Argentina": "Buenos Aires", 
        "Chile": "Santiago", 
        "Peru": "Lima", 
        "Colombia": "Bogotá"
        }

    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"The capital of {country} is {capital}"
    return "I don't know the capital of that country"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [calculator, web_search_mock]
tool_names = [tool.name for tool in tools]

prompt = hub.pull("hwchase17/react")
print(f"Prompt: {prompt.template}")

agent_chain = create_react_agent(llm, tools=tools, prompt=prompt, stop_sequence=False)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_chain, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3)

# print(agent_executor.invoke({"input": "What is the capital of Brazil?"}))
# print(agent_executor.invoke({"input": "What is the capital of Australia?"}))
print(agent_executor.invoke({"input": "How much is 10 + 10?"}))
