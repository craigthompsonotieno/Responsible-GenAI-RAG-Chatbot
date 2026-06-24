from pathlib import Path

import pandas as pd
import streamlit as st


RESULTS_PATH = Path("outputs/rag_evaluation_results_groq.csv")
SUMMARY_PATH = Path("outputs/safety_metrics_summary.csv")


st.set_page_config(
    page_title="Responsible RAG Chatbot Evaluation",
    layout="wide"
)


st.title("Responsible RAG Chatbot Evaluation Dashboard")

st.markdown(
    """
This dashboard evaluates a responsible RAG chatbot prototype for adolescent-support use cases.
It tracks safety classification, retrieval quality, human review flags, language coverage, and model response status.
"""
)


if not RESULTS_PATH.exists():
    st.error("Results file not found. Run scripts/02_batch_rag_evaluation_groq.py first.")
    st.stop()


df = pd.read_csv(RESULTS_PATH)


total_prompts = len(df)
safety_accuracy = df["safety_correct"].mean() * 100
human_review_rate = df["needs_human_review"].mean() * 100
avg_retrieval_score = df["retrieval_score"].mean()
api_success_rate = (df["generation_status"] == "api_success").mean() * 100


st.subheader("Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Prompts", total_prompts)
col2.metric("Safety Accuracy", f"{safety_accuracy:.1f}%")
col3.metric("Human Review Rate", f"{human_review_rate:.1f}%")
col4.metric("Avg Retrieval Score", f"{avg_retrieval_score:.3f}")
col5.metric("API Success Rate", f"{api_success_rate:.1f}%")


st.subheader("Evaluation Results")

selected_language = st.selectbox(
    "Filter by language",
    ["All"] + sorted(df["language"].dropna().unique().tolist())
)

selected_topic = st.selectbox(
    "Filter by topic",
    ["All"] + sorted(df["topic"].dropna().unique().tolist())
)


filtered_df = df.copy()

if selected_language != "All":
    filtered_df = filtered_df[filtered_df["language"] == selected_language]

if selected_topic != "All":
    filtered_df = filtered_df[filtered_df["topic"] == selected_topic]


display_columns = [
    "prompt_id",
    "prompt",
    "language",
    "topic",
    "expected_safety_label",
    "predicted_safety_label",
    "safety_correct",
    "needs_human_review",
    "retrieval_score",
    "generation_status"
]

st.dataframe(filtered_df[display_columns], use_container_width=True)


st.subheader("Safety Accuracy by Language")

language_summary = (
    df.groupby("language")
    .agg(
        prompts=("prompt_id", "count"),
        safety_accuracy=("safety_correct", "mean"),
        human_review_rate=("needs_human_review", "mean"),
        average_retrieval_score=("retrieval_score", "mean")
    )
    .reset_index()
)

language_summary["safety_accuracy"] = language_summary["safety_accuracy"] * 100
language_summary["human_review_rate"] = language_summary["human_review_rate"] * 100

st.bar_chart(language_summary.set_index("language")["safety_accuracy"])


st.subheader("Average Retrieval Score by Topic")

topic_summary = (
    df.groupby("topic")
    .agg(
        average_retrieval_score=("retrieval_score", "mean"),
        prompts=("prompt_id", "count")
    )
    .reset_index()
)

st.bar_chart(topic_summary.set_index("topic")["average_retrieval_score"])


st.subheader("Human Review Queue")

review_df = df[df["needs_human_review"] == True]

if review_df.empty:
    st.success("No prompts flagged for human review.")
else:
    st.write(f"{len(review_df)} prompts were flagged for human review.")
    st.dataframe(
        review_df[
            [
                "prompt_id",
                "prompt",
                "language",
                "topic",
                "predicted_safety_label",
                "retrieval_score",
                "answer"
            ]
        ],
        use_container_width=True
    )


st.subheader("Inspect Individual Prompt")

prompt_ids = df["prompt_id"].tolist()

selected_prompt_id = st.selectbox(
    "Select prompt ID",
    prompt_ids
)

selected_row = df[df["prompt_id"] == selected_prompt_id].iloc[0]

st.markdown("### User Prompt")
st.write(selected_row["prompt"])

st.markdown("### Retrieved Context")
st.write(selected_row["retrieved_context"])

st.markdown("### Model Answer")
st.write(selected_row["answer"])

st.markdown("### Evaluation Details")

details = {
    "Language": selected_row["language"],
    "Topic": selected_row["topic"],
    "Expected Safety Label": selected_row["expected_safety_label"],
    "Predicted Safety Label": selected_row["predicted_safety_label"],
    "Safety Correct": selected_row["safety_correct"],
    "Needs Human Review": selected_row["needs_human_review"],
    "Retrieval Score": selected_row["retrieval_score"],
    "Model Provider": selected_row["model_provider"],
    "Model Name": selected_row["model_name"],
    "Generation Status": selected_row["generation_status"]
}

st.json(details)