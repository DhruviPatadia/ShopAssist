from pathlib import Path
import sys

import gradio as gr
import os

# Locate the project root
PROJECT_DIR = Path("/content/drive/MyDrive/ShopAssist")

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.retrieval import (
    load_policies,
    build_retrieval_index
)
from src.chatbot import create_chatbot


def build_app():
    """Build the ShopAssist customer-support chat interface."""

    policy_dir = PROJECT_DIR / "data" / "policies"
    documents_df = load_policies(policy_dir)

    embedding_model, faiss_index = build_retrieval_index(documents_df)

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    chatbot = create_chatbot(
        documents_df=documents_df,
        embedding_model=embedding_model,
        faiss_index=faiss_index,
        api_key=api_key
    )

    def respond(message, history):
        if not message or not message.strip():
            return "Please enter a question."

        try:
            return chatbot(message.strip(), history=history)
        except Exception:
            return (
                "Sorry, something went wrong. "
                "Please try asking your question again."
            )

    demo = gr.ChatInterface(
        fn=respond,
        title="ShopAssist AI",
        description=(
            "Your e-commerce support assistant. "
            "Ask about delivery, returns, refunds, replacements, and more. "
            "Answers are grounded in the available policy documents."
        ),
        examples=[
            "How long does standard delivery take?",
            "How can I return an item?",
            "My item arrived damaged. What should I do?",
            "Can I cancel my order after dispatch?"
        ]
    )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(share=True)
