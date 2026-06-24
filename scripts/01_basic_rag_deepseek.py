import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found. Please check your .env file.")


client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)


KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_documents(folder_path):
    documents = []

    for file_path in folder_path.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        sections = text.split("\n\n")

        for i, section in enumerate(sections):
            clean_section = section.strip()

            if clean_section:
                documents.append(
                    {
                        "source": file_path.name,
                        "chunk_id": i + 1,
                        "text": clean_section
                    }
                )

    return pd.DataFrame(documents)


def retrieve_context(question, docs_df, top_k=2):
    vectorizer = TfidfVectorizer(stop_words="english")

    corpus = docs_df["text"].tolist()
    vectors = vectorizer.fit_transform(corpus + [question])

    doc_vectors = vectors[:-1]
    question_vector = vectors[-1]

    similarities = cosine_similarity(question_vector, doc_vectors).flatten()

    docs_df = docs_df.copy()
    docs_df["similarity"] = similarities

    top_docs = docs_df.sort_values("similarity", ascending=False).head(top_k)

    context = "\n\n".join(top_docs["text"].tolist())

    return context, top_docs


def classify_safety(question):
    question_lower = question.lower()

    needs_support_words = [
        "stressed", "stress", "anxious", "anxiety", "sad",
        "overwhelmed", "pressure", "worried", "scared"
    ]

    if any(word in question_lower for word in needs_support_words):
        return "needs_support"

    return "safe_general"


def generate_answer(question, context, safety_label):
    system_prompt = """
You are a safe, supportive assistant for adolescents.

Rules:
- Answer using only the provided context.
- Be brief, kind, and age-appropriate.
- Do not claim to be a doctor, therapist, lawyer, or emergency service.
- If the user seems stressed or needs support, encourage them to speak with a trusted adult, teacher, counselor, parent, guardian, or qualified professional.
- If the answer is not in the context, say you do not have enough approved information.
"""

    user_prompt = f"""
Safety label: {safety_label}

Approved context:
{context}

User question:
{question}

Write a helpful answer grounded in the approved context.
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


def run_rag(question):
    docs_df = load_documents(KNOWLEDGE_BASE_DIR)

    if docs_df.empty:
        raise ValueError("No documents found in data/knowledge_base.")

    context, top_docs = retrieve_context(question, docs_df)
    safety_label = classify_safety(question)
    answer = generate_answer(question, context, safety_label)

    result = {
        "question": question,
        "safety_label": safety_label,
        "retrieved_context": context,
        "answer": answer
    }

    results_df = pd.DataFrame([result])
    results_df.to_csv(OUTPUT_DIR / "rag_single_result.csv", index=False)

    print("\nQUESTION:")
    print(question)

    print("\nSAFETY LABEL:")
    print(safety_label)

    print("\nRETRIEVED CONTEXT:")
    print(context)

    print("\nANSWER:")
    print(answer)

    print("\nSaved to outputs/rag_single_result.csv")


if __name__ == "__main__":
    sample_question = "I feel stressed because of exams. What can I do?"
    run_rag(sample_question)