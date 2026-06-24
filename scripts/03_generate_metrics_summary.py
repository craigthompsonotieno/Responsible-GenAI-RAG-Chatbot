from pathlib import Path

import pandas as pd


INPUT_PATH = Path("outputs/rag_evaluation_results_groq.csv")
OUTPUT_PATH = Path("outputs/safety_metrics_summary.csv")


def main():
    df = pd.read_csv(INPUT_PATH)

    total_prompts = len(df)
    safety_accuracy = df["safety_correct"].mean()
    human_review_rate = df["needs_human_review"].mean()
    average_retrieval_score = df["retrieval_score"].mean()
    api_success_rate = (df["generation_status"] == "api_success").mean()

    summary = pd.DataFrame(
        [
            {
                "metric": "total_prompts",
                "value": total_prompts
            },
            {
                "metric": "safety_accuracy_percent",
                "value": round(safety_accuracy * 100, 1)
            },
            {
                "metric": "human_review_rate_percent",
                "value": round(human_review_rate * 100, 1)
            },
            {
                "metric": "average_retrieval_score",
                "value": round(average_retrieval_score, 3)
            },
            {
                "metric": "api_success_rate_percent",
                "value": round(api_success_rate * 100, 1)
            }
        ]
    )

    by_language = (
        df.groupby("language")
        .agg(
            prompts=("prompt_id", "count"),
            safety_accuracy=("safety_correct", "mean"),
            human_review_rate=("needs_human_review", "mean"),
            average_retrieval_score=("retrieval_score", "mean")
        )
        .reset_index()
    )

    by_language["safety_accuracy"] = round(by_language["safety_accuracy"] * 100, 1)
    by_language["human_review_rate"] = round(by_language["human_review_rate"] * 100, 1)
    by_language["average_retrieval_score"] = round(by_language["average_retrieval_score"], 3)

    by_topic = (
        df.groupby("topic")
        .agg(
            prompts=("prompt_id", "count"),
            safety_accuracy=("safety_correct", "mean"),
            human_review_rate=("needs_human_review", "mean"),
            average_retrieval_score=("retrieval_score", "mean")
        )
        .reset_index()
    )

    by_topic["safety_accuracy"] = round(by_topic["safety_accuracy"] * 100, 1)
    by_topic["human_review_rate"] = round(by_topic["human_review_rate"] * 100, 1)
    by_topic["average_retrieval_score"] = round(by_topic["average_retrieval_score"], 3)

    with pd.ExcelWriter(OUTPUT_PATH.with_suffix(".xlsx")) as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        by_language.to_excel(writer, sheet_name="by_language", index=False)
        by_topic.to_excel(writer, sheet_name="by_topic", index=False)
        df.to_excel(writer, sheet_name="full_results", index=False)

    summary.to_csv(OUTPUT_PATH, index=False)

    print("\nMetrics summary complete.")
    print(f"CSV saved to {OUTPUT_PATH}")
    print(f"Excel summary saved to {OUTPUT_PATH.with_suffix('.xlsx')}")
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()