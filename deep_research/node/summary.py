import logging
import os
from pathlib import Path
import re

from langchain_core.messages import HumanMessage 
from langchain_core.prompts.chat import SystemMessagePromptTemplate

from deep_research.common.agent_state import AgentState


class SummarizerNode:
    """
    A LangGraph node that summarizes search results (knowledges) based on
    the current search instruction and potentially previous history.
    """
    _PROMPT_FILE = Path(os.path.dirname(__file__)) / 'prompts' / 'summary.jinja'

    _REGEX = re.compile('<answer>(.*)</answer>', re.DOTALL)

    def __init__(self, llm):
        """
        Initializes the SummarizerNode with an LLM instance.

        Args:
            llm: The language model instance to use for summarization.
        """
        self.llm = llm
        try:
            self._system_prompt_template = SystemMessagePromptTemplate.from_template_file(
                self._PROMPT_FILE,
                template_format='jinja2',
                input_variables=[
                    'question',
                    'date',
                    'search_plan',
                    'knowledge', 
                    'history_result', 
                ],
            )
        except FileNotFoundError:
            logging.error(f"Summarizer prompt file not found at {self._PROMPT_FILE}")
            self._system_prompt_template = None

    def _generate_prompt(self, state: AgentState) -> list:
        """Generates the prompt for the LLM based on the current state."""
        if not self._system_prompt_template:
            raise FileNotFoundError(f"Summarizer prompt template file not loaded: {self._PROMPT_FILE}")

        knowledge_str = '\n'.join(state.get('knowledge', [])) if isinstance(state.get('knowledge'), list) else state.get('knowledge', '')
        history_result_str = state.get('history_result', '')

        system_prompt = self._system_prompt_template.format(
            question=state['question'],
            date=state['date'],
            search_plan=state['search_plan'],
            knowledge=knowledge_str,
            history_result=history_result_str,
        )

        user_prompt = HumanMessage(state['question'])

        return [system_prompt, user_prompt]

    def _invoke_llm(self, prompts):
        """Invokes the LLM with the generated prompt."""
        logging.info(f"Summarizer LLM Prompt: {prompts[0].content}")
        return self.llm.invoke(prompts)

    def _update_state_on_success(self, state: AgentState, response_content: str) -> AgentState:
        """Updates the agent state with the generated summary."""
        output = response_content.strip()
        logging.info(f"Generated summary: {output}...")

        matched = self._REGEX.search(output)
        if matched:
            final_answer = matched.groups()[0]

            return {**state, 'final_answer': final_answer, 'error': ''}
        return {**state, 'final_answer': '', 'error': 'Final Answer format error'}

    def _handle_error(self, state: AgentState, exception: Exception, context: str = "Summarization") -> AgentState:
        logging.error(f'Error in SummarizerNode ({context}): {exception}', exc_info=True)
        return {**state, 'history_result': "Failed to generate summary.", 'error': f'{context} failed: {exception}'}

    def __call__(self, state: AgentState) -> AgentState:
        """
        The main entry point for the LangGraph node.
        Executes the summarization based on the current state.
        """
        # Skip summarization if there were no keywords or search failed critically
        if not state.get('knowledges') and "Search execution failed" in state.get('error', ''):
            logging.warning("Skipping summarization due to previous search execution failure or no knowledges.")
            return {**state, 'history_result': "No information to summarize due to search failure."}
        
        try:
            prompt = self._generate_prompt(state)
            response = self._invoke_llm(prompt)
            return self._update_state_on_success(state, response.content.strip())
        except Exception as e:
            return self._handle_error(state, e, "LLM invocation or state update for summarization")
