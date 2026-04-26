import os
import json
from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Load your Gemini API Key
load_dotenv()

# 1. Define the Expected JSON Output Structure
class PostOutput(BaseModel):
    bot_id: str = Field(description="The name of the bot (e.g., Bot A)")
    topic: str = Field(description="The main topic of the post")
    post_content: str = Field(description="The highly opinionated, 280-character post")

# 2. Define the "Memory" for the LangGraph Workflow
class GraphState(TypedDict):
    bot_id: str
    persona: str
    search_query: str
    search_results: str
    final_post: dict

# 3. The Mock Search Tool
@tool
def mock_searxng_search(query: str) -> str:
    """Simulates a web search returning hardcoded headlines."""
    query = query.lower()
    if "crypto" in query or "bitcoin" in query:
        return "Bitcoin hits new all-time high amid regulatory ETF approvals."
    elif "ai" in query or "openai" in query:
        return "AI models show emergent reasoning capabilities, raising safety concerns."
    elif "market" in query or "rates" in query:
        return "Fed signals potential rate cuts by Q3; tech stocks rally."
    else:
        return "General global news reports stability in major sectors."

# Initialize the Gemini Model (Using 1.5 flash for structured output)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 4. Define the 3 Steps (Nodes) of our AI Agent
def decide_search_node(state: GraphState):
    """Step 1: The AI decides what to search for based on its personality."""
    
    # FIX: Split the prompt into "system" and "user" messages
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI bot with the following persona: {persona}."),
        ("user", "What specific keyword or short phrase do you want to search the news for today to post about? Output ONLY the search query (e.g., 'crypto', 'AI', 'markets').")
    ])
    chain = prompt | llm
    result = chain.invoke({"persona": state["persona"]})
    return {"search_query": result.content.strip()}

def web_search_node(state: GraphState):
    """Step 2: The AI uses the Search Tool to get real-world context."""
    results = mock_searxng_search.invoke({"query": state["search_query"]})
    return {"search_results": results}

def draft_post_node(state: GraphState):
    """Step 3: The AI reads the news and drafts a post in strict JSON format."""
    
    # FIX: Split the prompt into "system" and "user" messages
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI bot with the following persona: {persona}. You must output valid JSON matching the required schema. Set your bot_id as {bot_id}."),
        ("user", "Write a highly opinionated post (under 280 characters) reacting to this news: {news}")
    ])
    
    # This specifically forces Gemini to output the perfect JSON structure!
    structured_llm = llm.with_structured_output(PostOutput)
    chain = prompt | structured_llm
    
    result = chain.invoke({
        "persona": state["persona"],
        "news": state["search_results"],
        "bot_id": state["bot_id"]
    })
    
    # Convert Pydantic object to dictionary
    return {"final_post": result.model_dump() if hasattr(result, 'model_dump') else result.dict()}

# 5. Build the LangGraph Workflow (Connect the steps together)
workflow = StateGraph(GraphState)
workflow.add_node("decide_search", decide_search_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("draft_post", draft_post_node)

workflow.set_entry_point("decide_search")
workflow.add_edge("decide_search", "web_search")
workflow.add_edge("web_search", "draft_post")
workflow.add_edge("draft_post", END)

# Compile the brain!
app = workflow.compile()

# --- Test it ---
if __name__ == "__main__":
    print("Initializing LangGraph Agent Workflow...\n")
    
    # We are testing with Bot A (The Tech Maximalist)
    initial_state = {
        "bot_id": "Bot A",
        "persona": "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
        "search_query": "",
        "search_results": "",
        "final_post": {}
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    print("--- Final JSON Output ---")
    print(json.dumps(result["final_post"], indent=2))