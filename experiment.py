"""Experiment 1: reproducible TF-IDF text classification benchmark.

Student: Wang Yixin (王亦昕), 10245000462
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing train_data.csv and test_data_unlabeled.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for metrics, plots, model, and predictions.",
    )
    return parser.parse_args()


def vectorizer_configs() -> dict[str, dict]:
    return {
        "word_1gram": {
            "lowercase": True,
            "ngram_range": (1, 1),
            "min_df": 1,
            "max_df": 1.0,
            "max_features": 30_000,
            "sublinear_tf": False,
            "dtype": np.float64,
        },
        "word_1_2gram": {
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.98,
            "max_features": 80_000,
            "sublinear_tf": True,
            "dtype": np.float64,
        },
    }


def model_configs() -> list[tuple[str, float, object]]:
    configs: list[tuple[str, float, object]] = []
    for alpha in (0.1, 0.5, 1.0):
        configs.append(("MultinomialNB", alpha, MultinomialNB(alpha=alpha)))
    for c_value in (0.5, 1.0, 2.0):
        configs.append(
            (
                "LogisticRegression",
                c_value,
                LogisticRegression(
                    C=c_value,
                    max_iter=2_000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            )
        )
    for c_value in (0.5, 1.0, 2.0):
        configs.append(
            (
                "LinearSVM",
                c_value,
                LinearSVC(C=c_value, random_state=RANDOM_STATE, max_iter=5_000),
            )
        )
    return configs


def save_comparison_plot(results: pd.DataFrame, path: Path) -> None:
    plot_df = results.sort_values("macro_f1", ascending=True).copy()
    labels = [
        f"{row.model} ({row.parameter_name}={row.parameter_value:g}, {row.features})"
        for row in plot_df.itertuples()
    ]
    colors = {
        "MultinomialNB": "#2A9D8F",
        "LogisticRegression": "#E9C46A",
        "LinearSVM": "#E76F51",
    }
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    bars = ax.barh(
        labels,
        plot_df["macro_f1"],
        color=[colors[name] for name in plot_df["model"]],
    )
    ax.bar_label(bars, labels=[f"{value:.4f}" for value in plot_df["macro_f1"]], padding=3, fontsize=8)
    ax.set_xlim(max(0.0, plot_df["macro_f1"].min() - 0.08), 1.0)
    ax.set_xlabel("Validation macro-F1")
    ax.set_title("Model and hyperparameter comparison (seed = 42)")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_confusion_plot(matrix: np.ndarray, labels: np.ndarray, path: Path) -> None:
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(7.1, 6.1))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            value = normalized[i, j]
            if value >= 0.04 or i == j:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if value > 0.55 else "#152238")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Best model: row-normalized validation confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_per_class_plot(report: dict, labels: np.ndarray, path: Path) -> None:
    scores = [report[str(label)]["f1-score"] for label in labels]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    bars = ax.bar([str(x) for x in labels], scores, color="#457B9D")
    ax.bar_label(bars, labels=[f"{score:.3f}" for score in scores], padding=2, fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Class label")
    ax.set_ylabel("Validation F1")
    ax.set_title("Per-class F1 of the selected model")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.data_dir / "train_data.csv"
    test_path = args.data_dir / "test_data_unlabeled.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if list(train_df.columns) != ["text", "target"] or list(test_df.columns) != ["text"]:
        raise ValueError("Unexpected CSV schema; expected train=(text,target), test=(text).")

    texts = train_df["text"].astype(str)
    targets = train_df["target"].astype(int)
    labels = np.sort(targets.unique())
    x_train, x_valid, y_train, y_valid = train_test_split(
        texts,
        targets,
        test_size=0.20,
        stratify=targets,
        random_state=RANDOM_STATE,
    )

    rows: list[dict] = []
    fitted_candidates: dict[tuple[str, str, float], tuple[TfidfVectorizer, object]] = {}
    print(f"train={len(x_train)}, valid={len(x_valid)}, test={len(test_df)}, classes={len(labels)}")

    for feature_name, feature_params in vectorizer_configs().items():
        vectorizer = TfidfVectorizer(**feature_params)
        feature_start = time.perf_counter()
        train_matrix = vectorizer.fit_transform(x_train)
        valid_matrix = vectorizer.transform(x_valid)
        feature_seconds = time.perf_counter() - feature_start
        print(f"\n[{feature_name}] shape={train_matrix.shape}, vectorize={feature_seconds:.2f}s")

        for model_name, parameter_value, estimator in model_configs():
            parameter_name = "alpha" if model_name == "MultinomialNB" else "C"
            candidate = clone(estimator)
            fit_start = time.perf_counter()
            candidate.fit(train_matrix, y_train)
            fit_seconds = time.perf_counter() - fit_start
            predict_start = time.perf_counter()
            valid_predictions = candidate.predict(valid_matrix)
            predict_seconds = time.perf_counter() - predict_start
            train_predictions = candidate.predict(train_matrix)
            row = {
                "features": feature_name,
                "model": model_name,
                "parameter_name": parameter_name,
                "parameter_value": parameter_value,
                "vocabulary_size": len(vectorizer.vocabulary_),
                "train_accuracy": accuracy_score(y_train, train_predictions),
                "accuracy": accuracy_score(y_valid, valid_predictions),
                "macro_f1": f1_score(y_valid, valid_predictions, average="macro"),
                "weighted_f1": f1_score(y_valid, valid_predictions, average="weighted"),
                "vectorize_seconds": feature_seconds,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            }
            rows.append(row)
            fitted_candidates[(feature_name, model_name, parameter_value)] = (vectorizer, candidate)
            print(
                f"  {model_name:20s} {parameter_name}={parameter_value:<3g} "
                f"acc={row['accuracy']:.4f} macro-F1={row['macro_f1']:.4f} "
                f"fit={fit_seconds:.2f}s"
            )

    results = pd.DataFrame(rows).sort_values(
        ["macro_f1", "accuracy", "fit_seconds"], ascending=[False, False, True]
    )
    results.to_csv(args.output_dir / "validation_results.csv", index=False, float_format="%.6f")
    save_comparison_plot(results, args.output_dir / "model_comparison.png")

    best = results.iloc[0]
    best_key = (best["features"], best["model"], float(best["parameter_value"]))
    best_vectorizer, best_estimator = fitted_candidates[best_key]
    best_valid_matrix = best_vectorizer.transform(x_valid)
    best_predictions = best_estimator.predict(best_valid_matrix)
    best_report = classification_report(
        y_valid, best_predictions, labels=labels, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(best_report).T
    report_df.to_csv(args.output_dir / "best_classification_report.csv", float_format="%.6f")
    matrix = confusion_matrix(y_valid, best_predictions, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(args.output_dir / "confusion_matrix.csv")
    save_confusion_plot(matrix, labels, args.output_dir / "confusion_matrix.png")
    save_per_class_plot(best_report, labels, args.output_dir / "per_class_f1.png")

    error_mask = np.asarray(y_valid) != best_predictions
    errors = pd.DataFrame(
        {
            "row_index": x_valid.index[error_mask],
            "true_label": np.asarray(y_valid)[error_mask],
            "predicted_label": best_predictions[error_mask],
            "text_excerpt": x_valid.iloc[np.flatnonzero(error_mask)].str.replace(
                r"\s+", " ", regex=True
            ).str.slice(0, 350).values,
        }
    )
    errors.to_csv(args.output_dir / "misclassified_examples.csv", index=False)

    best_feature_params = vectorizer_configs()[str(best["features"])]
    best_model_template = next(
        model
        for name, value, model in model_configs()
        if name == best["model"] and value == float(best["parameter_value"])
    )
    final_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**best_feature_params)),
            ("classifier", clone(best_model_template)),
        ]
    )
    final_start = time.perf_counter()
    final_pipeline.fit(texts, targets)
    test_predictions = final_pipeline.predict(test_df["text"].astype(str))
    final_fit_seconds = time.perf_counter() - final_start
    pd.DataFrame(test_predictions).to_csv(
        args.output_dir / "predictions.csv", index=False, header=False
    )
    joblib.dump(final_pipeline, args.output_dir / "best_model.joblib", compress=3)

    classifier = final_pipeline.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        terms = np.asarray(final_pipeline.named_steps["tfidf"].get_feature_names_out())
        feature_rows = []
        for class_index, label in enumerate(classifier.classes_):
            top_indices = np.argsort(classifier.coef_[class_index])[-15:][::-1]
            for rank, index in enumerate(top_indices, start=1):
                feature_rows.append(
                    {
                        "label": int(label),
                        "rank": rank,
                        "term": terms[index],
                        "weight": classifier.coef_[class_index, index],
                    }
                )
        pd.DataFrame(feature_rows).to_csv(args.output_dir / "top_features.csv", index=False)

    summary = {
        "student": {"name": "王亦昕", "student_id": "10245000462"},
        "random_state": RANDOM_STATE,
        "split": {"train": len(x_train), "validation": len(x_valid), "test": len(test_df)},
        "best_validation": {
            "features": str(best["features"]),
            "model": str(best["model"]),
            "parameter_name": str(best["parameter_name"]),
            "parameter_value": float(best["parameter_value"]),
            "accuracy": float(best["accuracy"]),
            "macro_f1": float(best["macro_f1"]),
            "weighted_f1": float(best["weighted_f1"]),
        },
        "final_fit_seconds": final_fit_seconds,
        "prediction_count": int(len(test_predictions)),
        "prediction_label_counts": {
            str(int(label)): int(count)
            for label, count in pd.Series(test_predictions).value_counts().sort_index().items()
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nBest validation configuration:")
    print(json.dumps(summary["best_validation"], ensure_ascii=False, indent=2))
    print(f"Saved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
