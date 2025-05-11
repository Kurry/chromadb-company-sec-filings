import os
import chromadb
import logging
import warnings
import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools import Tool
from google.genai import types

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging with cleaner format
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Enable DEBUG logging only for essential modules
logging.getLogger('google.adk').setLevel(logging.INFO)
logging.getLogger('google.adk.tools').setLevel(logging.DEBUG)  # Keep tools debug for visibility

# Define constants
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "sec_db")
APP_NAME = "amzn_sec_agent"
USER_ID = "user"
SESSION_ID = "session"

# Local model configuration
MODEL_OLLAMA_PLUTUS = "ollama_chat/0xroyce/Plutus-3B:latest"  # Financial model

def chroma_search(query: str, n_results: int = 3):
    """Search ChromaDB for relevant documents about SEC filings.
    
    Args:
        query: The search query text related to SEC filings
        n_results: Number of results to return (default: 3)
    
    Returns:
        List of document texts from SEC filings
    """
    # Add debug logging
    logger = logging.getLogger(__name__)
    logger.info(f"ChromaDB tool called with query: {query}, n_results: {n_results}")
    
    try:
        # Explicitly convert n_results to int to avoid type issues
        n_results = int(n_results)
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collections = client.list_collections()
        if not collections:
            raise ValueError(f"No collections found in ChromaDB at: {CHROMA_DB_PATH}")
        
        # Get the first collection (assuming it's the SEC filings collection)
        collection_name = collections[0].name
        logger.info(f"Using collection: {collection_name}")
        collection = client.get_collection(collection_name)
        
        # Execute the query
        results = collection.query(query_texts=[query], n_results=n_results)
        logger.info(f"ChromaDB returned {len(results['documents'][0])} documents")
        
        return results["documents"][0]
    except Exception as e:
        logger.error(f"Error in ChromaDB search: {str(e)}")
        # Return an explanatory message rather than failing
        return [f"Error searching SEC filings database: {str(e)}"]

# Create a Tool instance for better ADK integration
chroma_search_tool = Tool(
    name="sec_filings_search",
    description="Search the SEC filings database for relevant information about companies",
    func=chroma_search
)

async def call_agent_async(runner, user_id, session_id, question):
    """Async function to call the agent with proper error handling.
    
    Args:
        runner: The ADK Runner instance
        user_id: User identifier
        session_id: Session identifier
        question: Question text
        
    Returns:
        The final response from the agent
    """
    logger = logging.getLogger(__name__)
    
    content = types.Content(role="user", parts=[types.Part(text=question)])
    final_response = ""
    
    try:
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response() and event.content and event.content.parts:
                final_response = event.content.parts[0].text
                break
    except Exception as e:
        logger.error(f"Error during agent run: {e}")
        final_response = f"An error occurred: {str(e)}"
    
    return final_response

def create_agent():
    """Create the SEC filings agent with the local Plutus model."""
    logger = logging.getLogger(__name__)
    
    # Use the local Ollama Plutus model
    logger.info("Setting up local Ollama/Plutus model...")
    model = LiteLlm(model=MODEL_OLLAMA_PLUTUS)
    
    # Create the agent with improved instructions
    agent = LlmAgent(
        model=model,
        name=APP_NAME,
        instruction="""
            You are a financial research assistant specializing in SEC filings analysis.
            
            CAPABILITIES:
            - Research and retrieve information from Amazon's SEC filings
            - Analyze financial data, business risks, and company strategies
            - Provide factual, citation-based answers grounded in official documents
            
            INSTRUCTIONS:
            1. ALWAYS use the sec_filings_search tool when asked about SEC filings or financial information
            2. Formulate specific, targeted queries when searching
            3. Include direct quotes with proper citations in your answers
            4. If multiple documents are relevant, synthesize the information
            5. If no relevant information is found, clearly state that fact
            
            Remember to only provide information that is directly supported by the SEC filings.
        """,
        tools=[chroma_search_tool],
    )
    
    return agent

def run_single_query():
    """Run a single query with the SEC filings agent."""
    # Create the agent
    root_agent = create_agent()
    
    # Set up ADK session and runner
    session_service = InMemorySessionService()
    session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)
    
    # Example question
    question = "What are the main business risks mentioned in Amazon's latest SEC filings?"
    print(f"\nQuestion: {question}\n")
    
    # Use asyncio to run the agent
    response = asyncio.run(call_agent_async(runner, USER_ID, SESSION_ID, question))
    
    # Display the response
    print("Agent Response:")
    print(response)

if __name__ == "__main__":
    # Check if ChromaDB exists
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"Error: ChromaDB not found at {CHROMA_DB_PATH}")
        print("Please run the sec_filings_db.py script first to create the database.")
        exit(1)
    
    # Run the example query
    print(f"Starting SEC Filings Agent with ChromaDB at: {CHROMA_DB_PATH}")
    run_single_query()
