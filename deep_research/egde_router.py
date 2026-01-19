from deep_research.common.agent_state import AgentState

def planner_route(state: AgentState):
    if state['current_step'] + 1 >= len(state['search_plan']):
        return 'summary'
    return 'search_keyword_generator'

def search_route(state: AgentState):
    if state['current_step'] + 1 >= len(state['search_plan']):
        return 'summary'
    return 'search_keyword_generator'
