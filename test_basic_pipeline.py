import pandas as pd


def test_result_files_exist():
    required_files = [
        "results/answer_quality_summary.csv",
        "results/oracle_context_summary.csv",
        "results/rag_generation_results.csv",
        "results/manual_faithfulness_review_labeled.csv",
    ]

    for file_path in required_files:
        df = pd.read_csv(file_path)
        assert len(df) > 0


def test_answer_quality_columns():
    df = pd.read_csv("results/answer_quality_summary.csv")

    required_columns = {"config_name", "em", "f1"}
    assert required_columns.issubset(set(df.columns))


def test_oracle_summary_columns():
    df = pd.read_csv("results/oracle_context_summary.csv")

    required_columns = {"config_name", "em", "f1"}
    assert required_columns.issubset(set(df.columns))


def test_manual_review_columns():
    df = pd.read_csv("results/manual_faithfulness_review_labeled.csv")

    required_columns = {
        "config_name",
        "question",
        "gold_answer",
        "generated_answer",
        "top_context",
        "faithfulness_label",
        "error_type",
    }

    assert required_columns.issubset(set(df.columns))