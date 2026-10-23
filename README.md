# ShopAssist AI — Intelligent E-commerce Customer Support Agent

ShopAssist AI is a retrieval-augmented customer support chatbot designed to answer common e-commerce questions using a collection of policy documents. It combines semantic search, vector retrieval, a large language model, and a Gradio chat interface to provide answers grounded in relevant policies.

## Project Overview

E-commerce customers often need quick, consistent answers about delivery, returns, refunds, replacements, payments, and other order-related questions.

ShopAssist retrieves relevant policy documents and uses them as context for generating responses. It is designed to cite its policy sources and acknowledge when the available documentation does not specify an answer.

## Key Features

- **Semantic search:** Uses Sentence Transformers to encode policy documents and customer queries.
- **Vector retrieval:** Uses FAISS to retrieve relevant policies by similarity.
- **Retrieval-Augmented Generation (RAG):** Supplies retrieved policy content to Gemini when generating answers.
- **Source-grounded responses:** Encourages answers to cite the relevant policy filenames.
- **Conversation context:** Passes recent chat history to the answer-generation function.
- **Interactive interface:** Provides a chat interface built with Gradio.
- **Retrieval tests:** Includes automated tests for policy loading, relevant-document retrieval, empty queries, and chatbot configuration.

## Architecture

```text
Customer question
       |
       v
Sentence Transformer
       |
       v
FAISS similarity search
       |
       v
Top relevant policy documents
       |
       v
Policy context + recent conversation
       |
       v
Gemini answer generation
       |
       v
Response with policy source citations
       |
       v
Gradio chat interface
```

### Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS |
| Data processing | pandas, NumPy |
| Language model | Gemini via `google-genai` |
| Chat interface | Gradio |
| Development environment | Google Colab |
| Version control | Git and GitHub |

## Repository Structure

```text
ShopAssist/
├── app/
│   └── gradio_app.py
├── data/
│   ├── policies/
│   └── evaluation/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_chatbot_app.ipynb
├── src/
│   ├── retrieval.py
│   └── chatbot.py
├── tests/
│   └── test_retrieval.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Knowledge Base

The knowledge base contains 15 Markdown policy documents covering:

- Account policies
- Cancellations
- Damaged and missing items
- Delivery and shipping
- Discounts and gift cards
- International shipping
- Payment methods and privacy
- Refunds, replacements, and returns
- Warranty information

The policies are stored in `data/policies/`.

## Retrieval Evaluation

The initial retrieval evaluation used a set of customer questions with expected relevant policy documents.

| Metric | Recorded result |
|---|---:|
| Mean Reciprocal Rank (MRR) | 0.9474 |
| Recall@3 | 100% |

These figures describe the earlier retrieval evaluation recorded during development. They are retrieval metrics, not measurements of overall answer correctness or customer satisfaction.

The project also includes `data/evaluation/rag_evaluation_results.csv` for reviewing previously recorded answer-generation evaluation results.

## Human-Reviewed Answer Evaluation

ShopAssist AI was evaluated using a benchmark of 20 customer questions covering returns, delivery, cancellations, international shipping, replacements, refunds, payments, discounts, missing items, warranties, gift cards, privacy, and an out-of-scope address-change question.

The benchmark compares generated responses with expected answers and relevant policy documents. Responses were reviewed using three criteria: answer correctness, policy-source support, and citation correctness.

### Preliminary Results

| Metric | Result |
|---|---:|
| Questions reviewed | 20 |
| Answers labelled correct | 20/20 (100%) |
| Answers marked as source-supported | 19/20 (95%) |
| Answers with uncertain source support | 1/20 (5%) |
| Citation accuracy among applicable labels | 19/19 (100%) |

**Interpretation:** These are preliminary, human-reviewed results based on the current benchmark labels. They should not be interpreted as independently validated model accuracy or as evidence of performance across all possible customer queries.

One question asked whether a delivery address could be changed after dispatch. The available policies did not specify whether this was possible, and the chatbot acknowledged the information gap rather than asserting an unsupported policy.

Citation accuracy excludes that question because its citation label was marked `not_applicable`. The international-shipping source was also reviewed against the policy documents, which confirm shipping to selected countries.

### Evaluation Files

The evaluation artifacts are stored in `data/evaluation/`:

- `evaluation_questions.csv` — benchmark questions, expected answers, and expected sources.
- `rag_evaluation_results.csv` — recorded retrieval and generated-answer results.
- `improved_evaluation_report.csv` — additional retrieval and phrase-check indicators.
- `human_review_benchmark.csv` — human-review labels and notes.
- `human_review_metrics.csv` — summary of the preliminary human-review metrics.

The benchmark is small and reflects a limited set of policy questions. Further evaluation should include more paraphrases, ambiguous queries, unsupported questions, and independent review of answer claims and citations.

## Automated Tests

The retrieval test suite is located at `tests/test_retrieval.py`.

The five recorded retrieval tests passed during development. They check:

1. Loading all 15 policy documents.
2. Confirming policy sources use Markdown filenames.
3. Retrieving the damaged-item policy for a damaged-item query.
4. Retrieving the delivery policy for a standard-delivery query.
5. Returning no results for an empty query.

A separate configuration check also passed for handling a missing Gemini API key.

Run the retrieval tests from the project root:

```bash
python tests/test_retrieval.py
```

The first run may download the embedding model if it is not cached.

## Setup and Usage

### 1. Install dependencies

Use Python in an environment compatible with the packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 2. Configure the Gemini API key

Set the `GEMINI_API_KEY` environment variable securely. Never commit API keys or other credentials to GitHub.

For example, in a terminal:

```bash
export GEMINI_API_KEY="your-api-key"
```

Use the equivalent environment-variable configuration for your operating system or deployment environment.

### 3. Launch the application

From the repository root:

```bash
python app/gradio_app.py
```

The application loads the policy documents, builds the retrieval index, and launches the Gradio interface. A public share URL may be generated by Gradio; such links are temporary and are not a permanent deployment.

### Example Questions

- How long does standard delivery take?
- How can I return an item?
- My item arrived damaged. What should I do?
- Can I change my delivery address after dispatch?

## Limitations

- Responses depend on the completeness and accuracy of the available policy documents.
- Semantic retrieval can return relevant but not always optimal documents.
- Source citations are generated as part of the model response and are not independently verified against every claim.
- Answer quality has not been established through a comprehensive, independently validated evaluation.
- Gemini API availability and rate limits may affect response generation.
- The current Gradio share link is intended for temporary testing, not permanent production hosting.

## Future Improvements

- Expand and strengthen automated evaluation of retrieval and answer correctness.
- Improve handling of follow-up questions and ambiguous customer requests.
- Validate citations and claims against retrieved policy text.
- Add more robust error handling, logging, and monitoring.
- Deploy the application to a persistent hosting environment.
- Add broader unit and integration tests.

## Project Status

The modular retrieval and chatbot code has been created, the Gradio application has launched successfully in Google Colab, and initial retrieval and manual chatbot checks have been completed.

---

Developed as an AI customer-support portfolio project.
