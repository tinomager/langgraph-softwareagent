import operator
from pydantic import BaseModel, Field
from typing import Annotated, List, Tuple
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults  
from langgraph.prebuilt import create_react_agent  
from typing import Union 
from typing import Literal  
from langgraph.graph import END  
from langgraph.graph import StateGraph, START  
import os
import asyncio
import logging

load_dotenv()

logging.basicConfig(
    filename="logfile.txt",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

  
system_prompt = """You are an expert software engineer and solution architect in 2024. 
    When given a technical problem:
    1. First create a plan to analyze the problem and research potential solutions
    2. Use the tools to gather information about best practices, patterns and existing solutions
    3. Finally, synthesize the collected information to propose a well-structured technical solution 
    that follows software engineering principles."""
  
llm = AzureChatOpenAI(model=os.getenv('GPT_4O_MODEL', 'gpt-4o-2'), azure_deployment=os.getenv('GPT_4O_MODEL', 'gpt-4o-2'))  
tools = [TavilySearchResults(max_results=3)]  
agent_executor = create_react_agent(llm, tools, prompt=system_prompt)  

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

planner_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a senior software architect working in 2024. For the given 
        technical challenge, create a systematic research and analysis plan.
        Break down complex problems into clear investigation steps. Each step should:
        - Focus on a specific technical aspect
        - Consider architectural implications
        - Research industry best practices and patterns
        - Evaluate potential solutions and trade-offs
        Make sure steps build on each other logically to arrive at a comprehensive solution.
        The final step should synthesize findings into concrete implementation recommendations."""),
        ("placeholder", "{messages}"),
    ]
)

planner = planner_prompt | AzureChatOpenAI(
    azure_deployment=os.getenv('GPT_4O_MINI_MODEL', 'gpt-4o-mini'), temperature=0, model=os.getenv('GPT_4O_MINI_MODEL', 'gpt-4o-mini')
).with_structured_output(Plan)

class Response(BaseModel):  
    """Response to user."""  
  
    response: str  
  
class Act(BaseModel):  
    """Action to perform."""  
  
    action: Union[Response, Plan] = Field(  
        description="Action to perform. If you want to respond to user, use Response. "  
                    "If you need to further use tools to get the answer, use Plan."  
    )  

replanner_prompt = ChatPromptTemplate.from_template(  
    """For the given software engineering challenge, review and adjust the analysis plan.
    Consider what we've learned so far and what additional technical aspects need investigation.
    
    Original problem:
    {input}
    
    Initial plan:
    {plan}
    
    Completed steps and findings:
    {past_steps}
    
    Determine next steps needed. Focus on:
    - Gaps in technical understanding
    - Architecture decisions that need validation
    - Implementation approaches to evaluate
    - Potential risks to mitigate
    
    Only include remaining steps needed. If sufficient information is gathered for a solution, 
    provide final recommendations."""  
    )  
  
replanner = replanner_prompt | AzureChatOpenAI(  
    model=os.getenv('GPT_4O_MODEL', 'gpt-4o-2'), temperature=0  
).with_structured_output(Act)  
  
async def execute_step(state: PlanExecute):  
    plan = state["plan"]  
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))  
    task = plan[0]  
    task_formatted = f"""For the following plan: {plan_str}\n\nYou are tasked with executing step {1}, {task}."""  
    agent_response = await agent_executor.ainvoke(  
        {"messages": [("user", task_formatted)]}  
    )  
    return {  
        "past_steps": [(task, agent_response["messages"][-1].content)],  
    }  
  
async def plan_step(state: PlanExecute):  
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})  
    return {"plan": plan.steps}  

async def replan_step(state: PlanExecute):  
    output = await replanner.ainvoke(state)  
    if isinstance(output.action, Response):  
        return {"response": output.action.response}  
    else:  
        return {"plan": output.action.steps}  
  
def should_end(state: PlanExecute):  
    if "response" in state and state["response"]:  
        return END  
    else:  
        return "agent"  
  
workflow = StateGraph(PlanExecute)  
  
# Add the plan node  
workflow.add_node("planner", plan_step)  
  
# Add the execution step  
workflow.add_node("agent", execute_step)  
  
# Add a replan node  
workflow.add_node("replan", replan_step)  
  
workflow.add_edge(START, "planner")  
  
# From plan we go to agent  
workflow.add_edge("planner", "agent")  
  
# From agent, we replan  
workflow.add_edge("agent", "replan")  
  
workflow.add_conditional_edges(  
    "replan",  
    # Next, we pass in the function that will determine which node is called next.  
    should_end,  
    ["agent", END],  
)  
  

app = workflow.compile()  

graph_filename = os.getenv("GRAPH_RENDER_FILE", "graph.png")
if not os.path.exists(graph_filename):
    graph = app.get_graph()
    graph.draw_mermaid_png(output_file_path=graph_filename)

inputs = {"input": "How can we build an ReAct agent that accepts inputs via a HTTP REST-API and streams the current progress via Websocket to the client? The Agent should be hosted in Azure Container Service. What steps to we need to take to implement a realiable and secure deployment?"}  

async def main():
    confic = { "recursion_limit": 30 }
    async for event in app.astream(inputs, config=confic):
        for k, v in event.items():
            if k != "__end__":
                print(v)
                logging.debug(v)

asyncio.run(main())