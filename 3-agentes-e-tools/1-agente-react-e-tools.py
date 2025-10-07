from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

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

prompt = PromptTemplate.from_template(
    """
    Answer the following questions as best you can.
    Only use the information you get from the tools, even if you know the answer.
    If the information is not provided by the tools, say you don't know.
    
    You have the following tools: {tools}
    
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action

    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Rules:
    - If you choose an Action, do NOT include Final Answer in the same step.
    - After Action and Action Input, stop and wait for Observation.
    - Never search the internet. Only use the tools provided.

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """
)

agent_chain = create_react_agent(llm, tools=tools, prompt=prompt, stop_sequence=False)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_chain, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors="Invalid format. Either provide an Action with Action Input or Final Answer only.",
    max_iterations=3)

# print(agent_executor.invoke({"input": "What is the capital of Brazil?"}))
print(agent_executor.invoke({"input": "What is the capital of Australia?"}))
# print(agent_executor.invoke({"input": "How much is 10 + 10?"}))
