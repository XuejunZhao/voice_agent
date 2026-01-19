import os
import logging
import re
from pathlib import Path

from langchain_core.prompts.chat import HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import HumanMessage

from deep_research.common.agent_state import AgentState


_PLAN_STEP_REGEX = re.compile(r'<max>(\d+)</max>\n<answer>(.+)</answer>', re.DOTALL)

_PROMPT_FILE = Path(os.path.dirname(__file__)) / 'prompts' / 'planner.jinja'

# Get a logger instance for this module
# logger = logging.getLogger(__name__)


class PlannerNode:
    """
    A LangGraph node that plans the search steps based on the user's question.
    """
    _PLAN_STEP_REGEX = re.compile(r'<max>(\d+)</max>\n<answer>(.+)</answer>', re.DOTALL)
    _PROMPT_FILE = Path(os.path.dirname(__file__)) / 'prompts' / 'planner.jinja'

    def __init__(self, llm):
        """
        Initializes the PlannerNode with an LLM instance.

        Args:
            llm: The language model instance to use for planning.
        """
        self.llm = llm
        # Pre-load the system prompt template for efficiency
        try:
            self._system_prompt_template = SystemMessagePromptTemplate.from_template_file(
                self._PROMPT_FILE,
                template_format='jinja2',
                input_variables=["date", "search_num"]
            )
        except FileNotFoundError:
            logging.error(f"Prompt file not found at {self._PROMPT_FILE}")
            self._system_prompt_template = None # Handle missing file gracefully

    def _generate_prompt(self, state: AgentState) -> list:
        """Generates the prompt for the LLM based on the current state."""
        if not self._system_prompt_template:
             raise FileNotFoundError(f"Prompt template file not loaded: {self._PROMPT_FILE}")

        system_prompt = self._system_prompt_template.format(date=state['date'], search_num=state['search_num'])
        user_prompt = HumanMessage(state['question'])
        return [system_prompt, user_prompt]

    def _invoke_llm(self, prompt: list):
        """Invokes the LLM with the generated prompt."""
        logging.info(f"Planner LLM Prompt: {prompt}")
        return self.llm.invoke(prompt)

    def _update_state_on_success(self, state: AgentState, response_content: str) -> AgentState:
        """Updates the agent state upon successful LLM invocation and parsing."""
        logging.info(f"LLM Response: {response_content}")
        matched = self._PLAN_STEP_REGEX.search(response_content)
        if matched:
            # Use group(index) to access captured groups
            max_steps_str = matched.group(1)
            plan_str = matched.group(2)
            return {
                **state,
                'max_steps': int(max_steps_str),
                'search_plan': plan_str.split('\n'),
                'current_step': 0,
                'error': '', # Clear previous errors on success
            }
        else:
            error_message = f'Planning format error: LLM response did not match expected pattern. Response: {response_content}'
            logging.warning(error_message) # Use warning for format errors
            return {
                **state,
                'max_steps': 0, # Indicate failure
                'search_plan': [], # Empty plan on failure
                'error': error_message,
            }

    def _handle_error(self, state: AgentState, exception: Exception, context: str = "Planning") -> AgentState:
        """Handles errors during the planning process and updates the state."""
        error_message = f'{context} failed: {exception}'
        logging.error(f'Error in PlannerNode ({context}): {exception}', exc_info=True) # Log exception details
        return {
            **state,
            'max_steps': 0, # Indicate failure
            'search_plan': [], # Empty plan on failure
            'error': error_message,
        }

    def __call__(self, state: AgentState) -> AgentState:
        """
        The main entry point for the LangGraph node.
        Orchestrates the planning of search steps.
        """
        try:
            prompt = self._generate_prompt(state)
            response = self._invoke_llm(prompt)
            return self._update_state_on_success(state, response.content.strip())
        except Exception as e:
            return self._handle_error(state, e, "LLM invocation or state update")
