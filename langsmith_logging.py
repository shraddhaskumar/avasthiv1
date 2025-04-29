import os
from langsmith import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up LangSmith Client
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
client = Client(api_url="https://api.smith.langchain.com", api_key=LANGCHAIN_API_KEY)

def log_chat_interaction(user_input, bot_response, sentiment, thread_id):
    """
    Logs chatbot interactions and metadata to LangSmith.
    """
    try:
        client.log_run(
            name="Avasthi Chatbot Run",
            inputs={"user_query": user_input, "thread_id": thread_id},
            outputs={"bot_response": bot_response, "predicted_sentiment": sentiment},
            metadata={"model": "gpt-4o-mini", "system": "Arohi AI"},
        )
        print("✅ LangSmith logging successful")
    except Exception as e:
        print(f"❌ LangSmith logging failed: {e}")
