# client/services/agent.py
import asyncio
import json
import urllib.parse

from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.providers.openai import OpenAIProvider
from httpx import AsyncClient
from starlette.requests import Request

from client.config import logger


class AgentService:
    """Service for handling interactions between web UI, MCP server, and LLM providers."""
    
    def __init__(self, mcp_url, database_url, llm, api_key):
        """
        Initialize the agent service.
        
        Args:
            mcp_url: URL for the MCP server
            database_url: PostgreSQL connection string
            llm: LLM provider name (Anthropic, Gemini, or OpenAI)
            api_key: API key for the LLM provider
        """
        self.mcp_url = mcp_url
        self.database_url = database_url
        self.llm_name = llm
        self.api_key = api_key
        self.conn_id = None
        self.session = None
        self.agent = self._create_agent(llm, api_key)
        
    def _create_agent(self, llm_name, api_key):
        """Create an agent with the appropriate LLM provider."""
        custom_http_client = AsyncClient(timeout=30)
        
        if llm_name == "Anthropic":
            model = AnthropicModel(
                'claude-3-7-sonnet-20250219',
                provider=AnthropicProvider(api_key=api_key, http_client=custom_http_client),
            )
        elif llm_name == "Gemini":
            model = GeminiModel(
                'gemini-2.0-flash',
                provider=GoogleGLAProvider(api_key=api_key, http_client=custom_http_client),
            )
        elif llm_name == "Open AI":
            model = OpenAIModel(
                'gpt-4o-mini',
                provider=OpenAIProvider(api_key=api_key, http_client=custom_http_client),
            )
        else:
            # Default to Claude
            model = AnthropicModel(
                'claude-3-5-sonnet-20240620',
                provider=AnthropicProvider(api_key=api_key, http_client=custom_http_client),
            )
            
        return Agent(model)
    
    async def initialize(self):
        """
        Initialize the connection to the MCP server and database.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Connecting to MCP server at {self.mcp_url}")
            
            # Create SSE client and store context manager for proper cleanup
            self._sse_client = sse_client(url=self.mcp_url)
            self._streams = await self._sse_client.__aenter__()
            
            # Create and initialize the session
            self._session_client = ClientSession(*self._streams)
            self.session = await self._session_client.__aenter__()
            
            # Initialize the session
            await self.session.initialize()
            logger.info("MCP session initialized")
            
            # Connect to database
            if not self.database_url:
                logger.error("No database URL provided")
                return False
            
            connect_result = await self.session.call_tool(
                "connect",
                {"connection_string": self.database_url}
            )
            
            # Extract conn_id from the response
            if hasattr(connect_result, 'content') and connect_result.content:
                content = connect_result.content[0]
                if hasattr(content, 'text'):
                    try:
                        result_data = json.loads(content.text)
                        self.conn_id = result_data.get('conn_id')
                        logger.info(f"Connected to database with ID: {self.conn_id}")
                        return True
                    except json.JSONDecodeError:
                        logger.error(f"Error parsing connect result: {content.text[:100]}")
            
            logger.error("Failed to establish database connection")
            return False
            
        except Exception as e:
            logger.error(f"Error initializing agent service: {str(e)}")
            return False
    
    async def close(self):
        """Close the connection to the MCP server and database."""
        try:
            if self.conn_id and self.session:
                # Try to disconnect from database
                try:
                    await self.session.call_tool(
                        "disconnect",
                        {"conn_id": self.conn_id}
                    )
                    logger.info(f"Disconnected from database with ID: {self.conn_id}")
                except Exception as e:
                    logger.error(f"Error disconnecting from database: {str(e)}")
            
            # Properly exit the context managers in reverse order
            if hasattr(self, '_session_client') and self._session_client:
                try:
                    await self._session_client.__aexit__(None, None, None)
                    logger.info("Closed MCP session")
                except Exception as e:
                    logger.error(f"Error closing session: {str(e)}")
            
            if hasattr(self, '_sse_client') and self._sse_client:
                try:
                    await self._sse_client.__aexit__(None, None, None)
                    logger.info("Closed SSE connection")
                except Exception as e:
                    logger.error(f"Error closing SSE connection: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error during close: {str(e)}")
            
    def _clean_sql_query(self, sql_query):
        """
        Clean a SQL query by handling escaped quotes and trailing backslashes.
        
        Args:
            sql_query: The SQL query to clean
            
        Returns:
            str: Cleaned SQL query
        """
        # Use unicode_escape to properly handle all escape sequences
        import codecs
        
        # Handle as unicode escape sequence
        result = codecs.decode(sql_query, 'unicode_escape')
        
        # Remove any extraneous whitespace or newlines
        result = result.strip()
        
        return result
        
    async def process_query(self, user_query):
        """
        Process a natural language query, generate SQL, and execute it.
        
        Args:
            user_query: The natural language query
            
        Returns:
            dict: Result containing generated SQL, explanation, and query results
        """
        if not self.conn_id:
            return {
                "success": False,
                "error": "Not connected to database",
                "sql": "",
                "explanation": "",
                "results": []
            }
            
        try:
            # Get the prompt from server using new generate_sql prompt
            # We only need to send conn_id and query - server will fetch schema internally
            logger.info("Getting SQL generation prompt with server-side schema fetching")
            prompt_response = await self.session.get_prompt('generate_sql', {
                'conn_id': self.conn_id,
                'nl_query': user_query
            })
            
            # Extract messages from prompt response
            if not hasattr(prompt_response, 'messages') or not prompt_response.messages:
                return {
                    "success": False,
                    "error": "Invalid prompt response from server",
                    "sql": "",
                    "explanation": "",
                    "results": []
                }
            
            # Prepare messages for the LLM
            messages = []
            for msg in prompt_response.messages:
                messages.append({
                    "role": msg.role,
                    "content": msg.content.text if hasattr(msg.content, 'text') else str(msg.content)
                })
            
            # Use the agent to generate SQL
            logger.info("Sending query to LLM for SQL generation")
            llm_response = await self.agent.run(str(messages))
            
            # Extract SQL from the response
            response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            # Extract SQL from response
            sql_query = None
            
            # Look for SQL in code blocks
            if "```sql" in response_text and "```" in response_text.split("```sql", 1)[1]:
                sql_start = response_text.find("```sql") + 6
                remaining_text = response_text[sql_start:]
                sql_end = remaining_text.find("```")
                
                if sql_end > 0:
                    sql_query = remaining_text[:sql_end].strip()
            
            # If still no SQL query found, check if the whole response might be SQL
            if not sql_query and ("SELECT" in response_text or "WITH" in response_text):
                for keyword in ["WITH", "SELECT", "CREATE", "INSERT", "UPDATE", "DELETE"]:
                    if keyword in response_text:
                        keyword_pos = response_text.find(keyword)
                        sql_query = response_text[keyword_pos:].strip()
                        for end_marker in ["\n\n", "```"]:
                            if end_marker in sql_query:
                                sql_query = sql_query[:sql_query.find(end_marker)].strip()
                        break
            
            if not sql_query:
                return {
                    "success": False,
                    "error": "Could not extract SQL from the LLM response",
                    "sql": "",
                    "explanation": response_text,
                    "results": []
                }
            
            # Clean the SQL query
            sql_query = self._clean_sql_query(sql_query)
            
            # Add trailing semicolon if missing
            if not sql_query.endswith(';'):
                sql_query = sql_query + ';'
                
            # Execute the query
            logger.info(f"Executing SQL query: {sql_query}")
            query_result = await self.session.call_tool(
                "pg_query",
                {
                    "query": sql_query,
                    "conn_id": self.conn_id
                }
            )
            
            # Process results
            results = []
            if hasattr(query_result, 'content') and query_result.content:
                # Extract all content items and parse the JSON
                for content_item in query_result.content:
                    if hasattr(content_item, 'text'):
                        try:
                            # Parse each row from JSON
                            row_data = json.loads(content_item.text)
                            if isinstance(row_data, list):
                                results.extend(row_data)
                            else:
                                results.append(row_data)
                        except json.JSONDecodeError:
                            logger.error(f"Error parsing result item: {content_item.text[:100]}")
            
            # Generate explanation based on SQL
            explanation = f"Generated SQL to answer your query about '{user_query}'."
            
            return {
                "success": True,
                "sql": sql_query,
                "explanation": explanation,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "sql": "",
                "explanation": "",
                "results": []
            }
    
    @classmethod
    async def from_request(cls, request):
        """
        Create an AgentService instance from a Starlette request.
        
        Args:
            request: The Starlette request object
            
        Returns:
            AgentService: An initialized AgentService instance
        """
        # Get settings from session
        llm = request.session.get('LLM', 'Anthropic')
        api_key = request.session.get('LLM_API_KEY', '')
        mcp_url = request.session.get('PG_MCP_SERVER_URL', '')
        database_url = request.session.get('DATABASE_URL', '')
        
        if not api_key:
            raise ValueError("LLM API key is required")
        
        if not mcp_url:
            raise ValueError("MCP server URL is required")
        
        if not database_url:
            raise ValueError("Database URL is required")
            
        # Create the service (no initialization needed now)
        service = cls(mcp_url, database_url, llm, api_key)
        await service.initialize()
        
        return service