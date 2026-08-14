"""
RAG Evaluation Script using Ragas (v0.3.9 API - stable)

This script tests the RAG system against a set of known question/answer pairs
(based on test.pdf - the BrightLeaf Technologies document) and measures:

- Faithfulness: Is the answer grounded in the retrieved context (no hallucination)?
- Answer Relevancy: Does the answer actually address the question?
- Context Precision: Are the retrieved chunks actually relevant to the question?

Run this AFTER uploading test.pdf to your running app (local or deployed).
"""

import os
import sys
import types
import requests
from dotenv import load_dotenv

load_dotenv()

# ---- WORKAROUND ----
# Some versions of ragas try to import ChatVertexAI from a path that no longer
# exists in newer langchain-community versions. We don't use VertexAI at all,
# so we inject a harmless stub module to satisfy that import and avoid the crash.
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except ModuleNotFoundError:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # placeholder, never actually used
        pass

    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

# ---- CONFIG ----
API_BASE_URL = "http://127.0.0.1:8080"  # change to your Render URL to test the live deployment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# ---- TEST SET ----
test_cases = [
    {
        "question": "When was BrightLeaf Technologies founded and by whom?",
        "ground_truth": "BrightLeaf Technologies was founded in 2016 by Ananya Sharma."
    },
    {
        "question": "Where is the company headquarters located?",
        "ground_truth": "The company headquarters is located in Pune, Maharashtra, India."
    },
    {
        "question": "How many employees does the company have?",
        "ground_truth": "The company has 342 employees."
    },
    {
        "question": "What was the total revenue in fiscal year 2025?",
        "ground_truth": "The total revenue in fiscal year 2025 was 18.4 million dollars."
    },
    {
        "question": "Who is the CTO and when did they join?",
        "ground_truth": "Rohan Verma is the CTO and joined the company in 2019."
    },
    {
        "question": "What products does BrightLeaf Technologies make?",
        "ground_truth": "BrightLeaf Technologies makes LeafCRM, LeafPay, and LeafDesk."
    },
    {
        "question": "What are the company's future plans?",
        "ground_truth": "The company plans to launch LeafAnalytics in 2027 and open a Singapore office by 2028."
    },
]


def get_answer_from_api(question: str):
    """Calls the running FastAPI /chat endpoint and returns the answer + retrieved contexts."""
    response = requests.post(
        f"{API_BASE_URL}/chat",
        json={"question": question},
        timeout=60
    )
    response.raise_for_status()
    data = response.json()

    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")

    answer = data["answer"]
    contexts = [s["snippet"] for s in data["sources"]]
    return answer, contexts


def build_dataset():
    from datasets import Dataset

    questions, answers, contexts_list, ground_truths = [], [], [], []

    print(f"Running {len(test_cases)} test questions against {API_BASE_URL} ...\n")

    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        print(f"[{i}/{len(test_cases)}] {question}")

        answer, contexts = get_answer_from_api(question)

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(case["ground_truth"])

        print(f"   -> {answer}\n")

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })


def run_evaluation():
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    dataset = build_dataset()

    from ragas.run_config import RunConfig

    # Wrap Groq and HF embeddings so Ragas can use them as the "judge"
    eval_llm = LangchainLLMWrapper(
        ChatGroq(groq_api_key=GROQ_API_KEY, model_name="openai/gpt-oss-20b")
    )
    eval_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=HF_TOKEN
        )
    )

    # Groq's API rejects n > 1 (multiple completions per call), but answer_relevancy
    # normally asks for 3 generations at once. Force strictness to 1 so it only asks for one.
    answer_relevancy.strictness = 1

    # Slow down and space out requests so we don't hit Groq's free-tier rate limits,
    # and give each call more time before giving up.
    run_config = RunConfig(timeout=120, max_workers=2)

    print("Running Ragas evaluation (this calls the LLM several times, may take a minute)...\n")

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config,
    )

    print("\n===== RAGAS EVALUATION RESULTS =====\n")
    df = results.to_pandas()
    print("Columns returned:", df.columns.tolist())
    print()
    print(df.to_string(index=False))

    print("\n----- AVERAGE SCORES -----")
    for col in ["faithfulness", "answer_relevancy", "context_precision"]:
        if col in df.columns:
            print(f"{col}: {df[col].mean(skipna=True):.2f}")
        else:
            print(f"{col}: column not found in results")

    df.to_csv("ragas_results.csv", index=False)
    print("\nFull results saved to ragas_results.csv")


if __name__ == "__main__":
    run_evaluation()