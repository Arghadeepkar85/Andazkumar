import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables (Make sure GOOGLE_API_KEY is in your .env file)
load_dotenv()

# 1. Setup the Vector Database using Google's Embedding Model
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = Chroma(embedding_function=embeddings_model)

# Define and store the personas
personas = {
    "Bot A": "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
    "Bot B": "I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature.",
    "Bot C": "I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI."
}

# Add personas to ChromaDB
vector_store.add_texts(
    texts=list(personas.values()),
    metadatas=[{"bot_id": name} for name in personas.keys()]
)

# 2. The Execution Function
def route_post_to_bots(post_content: str, threshold: float = 0.50):
    """Routes a post to bots based on similarity."""
    
    # Perform similarity search with scores
    results = vector_store.similarity_search_with_relevance_scores(post_content, k=3)
    
    matched_bots = []
    for doc, score in results:
        # Note: Different embedding models scale scores differently. 
        # We use a threshold of 0.50 here for testing purposes, but you can tweak it to 0.85 if needed.
        if score > threshold:
            matched_bots.append({
                "bot_id": doc.metadata["bot_id"],
                "score": round(score, 4)
            })
            
    return matched_bots

# --- Test it ---
if __name__ == "__main__":
    test_post = "OpenAI just released a new model that might replace junior developers."
    print("Incoming Post:", test_post)
    
    matches = route_post_to_bots(test_post, threshold=0.10)
    print("\nMatched Bots:")
    for match in matches:
        print(f"- {match['bot_id']} (Score: {match['score']})")