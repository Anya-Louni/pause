"""Check that .env loads and one Gemini call and one Tavily call work."""

import os
import sys

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch


def main() -> int:
    load_dotenv()
    missing = [k for k in ("GOOGLE_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
    if missing:
        print(f"Missing keys: {', '.join(missing)}")
        return 1

    model_name = os.getenv("MODEL", "google_genai:gemini-3.1-flash-lite")
    reply = init_chat_model(model_name).invoke("Reply with the single word: ready")
    print(f"model {model_name}: {reply.text.strip()}")

    results = TavilySearch(max_results=1).invoke({"query": "LangChain create_agent"})
    top = results["results"][0]
    print(f"tavily: {top['title']} ({top['url']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
