import os
import logging
import re
from pathlib import Path

from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage

from deep_research.common.agent_state import AgentState

# Get a logger instance for this module
# logger = logging.getLogger(__name__)

class SearchKeywordGeneratorNode:
    """
    A LangGraph node that generates search keywords based on the current search plan step
    and previous context (knowledge, history).
    """
    # Regex to extract keywords, assuming they are comma-separated or in a list format
    # This might need adjustment based on the LLM's typical output for keywords.
    _KEYWORDS_REGEX = re.compile(r'(?:<answer>(.*)</answer>)?.*<search>(.*)</search>', re.DOTALL)
    _PROMPT_FILE = Path(os.path.dirname(__file__)) / 'prompts' / 'search.jinja'

    def __init__(self, llm):
        """
        Initializes the SearchKeywordGeneratorNode with an LLM instance.

        Args:
            llm: The language model instance to use for generating search keywords.
        """
        self.llm = llm
        try:
            self._system_prompt_template = SystemMessagePromptTemplate.from_template_file(
                self._PROMPT_FILE, 
                template_format='jinja2',
                input_variables=[
                    'date',
                    'search_plan',
                    'current_step',
                    'knowledge', 
                    'history_result', 
                ],
            )
        except FileNotFoundError:
            logging.error(f"Search prompt file not found at {self._PROMPT_FILE}")
            self._system_prompt_template = None

    def _generate_prompt(self, state: AgentState) -> list:
        """Generates the prompt for the LLM based on the current state."""
        if not self._system_prompt_template:
            raise FileNotFoundError(f"Search prompt template file not loaded: {self._PROMPT_FILE}")

        # Ensure knowledges and history_result are strings, handle if they are lists or None
        knowledge_str = '\n'.join(state.get('knowledge', [])) if isinstance(state.get('knowledge'), list) else state.get('knowledge', '')
        history_result_str = state.get('history_result', '')

        state['current_step'] += 1

        system_prompt = self._system_prompt_template.format(
            date=state['date'],
            search_plan=state['search_plan'],
            current_step=state['current_step'],
            knowledge=knowledge_str,
            history_result=history_result_str,
        )

        user_prompt = HumanMessage(state['question'])

        return [system_prompt, user_prompt]

    def _invoke_llm(self, prompt: list):
        """Invokes the LLM with the generated prompt."""
        logging.info(f"Search Keyword Generator LLM Prompt: {prompt}")
        return self.llm.invoke(prompt)

    def _update_state_on_success(self, state: AgentState, response_content: str) -> AgentState:
        """Updates the agent state with generated search keywords."""
        logging.info(f"LLM Response: {response_content}")
        matched = self._KEYWORDS_REGEX.search(response_content)
        search_keywords = []
        summary = None
        if matched is not None:
            groups = matched.groups()
            summary = groups[0]
            search_keywords = [kw.strip() for kw in groups[1].split('|') if kw.strip()]
                
            history_result = state.get('history_result', [])
            if summary is not None:
                history_result.append(summary)
        
            logging.info(f"Generated search keywords: {search_keywords}")
            return {**state, 'search_keywords': search_keywords, 'history_result': history_result, 'error': ''}
        else:
            logging.warning('LLM bad response format, response: %s', response_content)
            return {**state, 'search_keywords': [], 'error': 'LLM response format error'}


    def _handle_error(self, state: AgentState, exception: Exception, context: str = "Search Keyword Generation") -> AgentState:
        logging.error(f'Error in SearchKeywordGeneratorNode ({context}): {exception}', exc_info=True)
        return {**state, 'search_keywords': [], 'error': f'{context} failed: {exception}'}

    def __call__(self, state: AgentState) -> AgentState:
        try:
            prompt = self._generate_prompt(state)
            response = self._invoke_llm(prompt)
            return self._update_state_on_success(state, response.content.strip())
        except Exception as e:
            return self._handle_error(state, e, "LLM invocation or state update for search keywords")
