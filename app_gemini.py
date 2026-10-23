
from pathlib import Path
import os

import gradio as gr

from src.retrieval_gemini import load_policies, build_retrieval_index
from src.chatbot_gemini import create_chatbot

PROJECT_DIR = Path(__file__).resolve().parent


def build_app():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in the hosting environment.")

    policy_dir = PROJECT_DIR / "data" / "policies"
    documents_df = load_policies(policy_dir)

    embedding_client, faiss_index = build_retrieval_index(
        documents_df, api_key=api_key
    )

    chatbot = create_chatbot(
        documents_df=documents_df,
        embedding_client=embedding_client,
        faiss_index=faiss_index,
        api_key=api_key
    )

    def respond(message, history):
        if not message or not message.strip():
            return "Please enter a question."
        return chatbot(message.strip(), history=history)

    return gr.ChatInterface(
        fn=respond,
        title="ShopAssist AI",
        description=(
            "Your e-commerce support assistant. Ask about delivery, returns, "
            "refunds, replacements, and more. Answers use the available policies."
        ),
        examples=[
            "How long does standard delivery take?",
            "How can I return an item?",
            "My item arrived damaged. What should I do?",
            "Can I cancel my order after dispatch?"
        ]
    )


if __name__ == "__main__":
    app = build_app()
    port = int(os.environ.get("PORT", "10000"))
    app.launch(server_name="0.0.0.0", server_port=port)
