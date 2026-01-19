from functools import partial

from langgraph.graph import StateGraph, END, START

from deep_research.common.agent_state import AgentState
from deep_research.egde_router import search_route, planner_route
from deep_research.node.planner import PlannerNode
from deep_research.node.search_keyword_generator import SearchKeywordGeneratorNode
from deep_research.node.search_executor import SearchExecutorNode
from deep_research.node.summary import SummarizerNode


def create_agent(llm, search_engine):

    workflow = StateGraph(AgentState)
    planner = PlannerNode(llm=llm)
    search_keyword_generator = SearchKeywordGeneratorNode(llm=llm)
    search_executor = SearchExecutorNode(search_tool=search_engine)
    summary = SummarizerNode(llm=llm)


    workflow.add_node('planner', planner)
    workflow.add_node('search_keyword_generator', search_keyword_generator)
    workflow.add_node('search_executor', search_executor)
    workflow.add_node('summary', summary)

    workflow.add_edge(START, 'planner')

    workflow.add_conditional_edges('planner', planner_route)

    workflow.add_edge('search_keyword_generator', 'search_executor')

    workflow.add_conditional_edges('search_executor', search_route)

    workflow.add_edge('summary', END)

    agent = workflow.compile()

    return agent
