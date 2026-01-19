from typing import List, TypedDict

class AgentState(TypedDict):
    question: str
    date: str
    search_num: int

    max_steps:int
    search_plan: List[str]
    current_step: int

    search_keywords: List[str]
    knowledge: List[str] 
    history_result: List[str]

    final_answer: str
    
    error: str
