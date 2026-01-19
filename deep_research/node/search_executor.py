import logging
from typing import List

from langchain_core.tools import BaseTool

from deep_research.common.agent_state import AgentState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchExecutorNode:
    """
    A LangGraph node that executes a search using a provided search tool
    based on the search keywords in the agent state.
    """

    def __init__(self, search_tool):
        """
        Initializes the SearchExecutorNode with a search tool instance.

        Args:
            search_tool: An instance of SearxSearchWrapper for searching.
        """
        self.search_tool = search_tool
        logger.info(f"SearchExecutorNode initialized with search_tool type: {type(search_tool)}")

    def _execute_search(self, keywords: List[str]) -> List[List[dict]]:
        """Executes the search using the configured tool."""
        logger.info(f"🔍 Executing search for keywords: {keywords}")
        results = []
        for keyword in keywords:
            try:
                logger.info(f"  Searching for: '{keyword}'")
                # SearxSearchWrapper uses run() method, not results()
                # It returns a list of dictionaries with 'title', 'snippet', 'link'
                search_result = self.search_tool.run(keyword)
                logger.info(f"  Search result type: {type(search_result)}, length: {len(search_result) if isinstance(search_result, list) else 'N/A'}")
                
                # Convert to list of dicts if needed
                if isinstance(search_result, str):
                    # If it returns a string, try to parse it
                    logger.warning(f"  Search returned string instead of list: {search_result[:100]}")
                    # Create a dict from the string result
                    results.append([{'title': keyword, 'snippet': search_result, 'link': ''}])
                elif isinstance(search_result, list):
                    results.append(search_result)
                else:
                    logger.warning(f"  Unexpected search result type: {type(search_result)}")
                    results.append([])
            except Exception as e:
                logger.error(f"  ❌ Search failed for keyword '{keyword}': {e}", exc_info=True)
                results.append([])
        
        logger.info(f"🔍 Total search results: {len(results)} keyword searches, {sum(len(r) for r in results)} total results")
        return results

    def _update_state_on_success(self, state: AgentState, search_results: List[List[dict]]) -> AgentState:
        """Updates the agent state with the search results."""
        logger.info(f'📊 Processing {len(search_results)} keyword search result sets')
        # TODO: dedup
        knowledge = []
        for sublist in search_results:
            for result in sublist:
                if isinstance(result, dict):
                    if 'title' in result and 'snippet' in result:
                        knowledge.append(f"{result['title']}\n{result['snippet']}\n")
                    elif 'content' in result:
                        # Handle different result formats
                        knowledge.append(f"{result.get('title', 'Result')}\n{result['content']}\n")
        
        logger.info(f"📚 Extracted {len(knowledge)} knowledge items from search results")
        if knowledge:
            logger.info(f"📝 First knowledge item preview: {knowledge[0][:200]}...")
        else:
            logger.warning("⚠️  No knowledge extracted from search results!")
            logger.debug(f"Search results structure: {search_results}")
        
        return {**state, 'knowledge': knowledge, 'error': ''}

    def _handle_error(self, state: AgentState, exception: Exception, context: str = "Search Execution") -> AgentState:
        logging.error(f'Error in SearchExecutorNode ({context}): {exception}', exc_info=True)
        return {**state, 'knowledge': [], 'error': f'{context} failed: {exception}'}

    def __call__(self, state: AgentState) -> AgentState:
        """
        The main entry point for the LangGraph node.
        Executes the search based on the state's search_keywords.
        """
        try:
            keywords = state.get('search_keywords', [])
            logger.info(f"🔎 SearchExecutorNode called with {len(keywords)} keywords: {keywords}")
            if not keywords:
                logger.warning("⚠️  No search keywords provided!")
                return {**state, 'knowledge': [], 'error': 'No search keywords provided'}
            
            search_results = self._execute_search(keywords)
            return self._update_state_on_success(state, search_results=search_results)
        except Exception as e:
            logger.error(f"❌ Search execution error: {e}", exc_info=True)
            return self._handle_error(state, e, "Search execution")
