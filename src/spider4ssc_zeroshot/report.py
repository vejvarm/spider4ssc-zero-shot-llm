from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LANGUAGE_ORDER = ["sql", "sparql", "cypher"]
REPORT_COLUMNS = [
    "run_id",
    "model_id",
    "language",
    "execution_accuracy",
    "n_examples",
    "n_empty_predictions",
]


def collect_scores(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for scores_file in sorted(runs_dir.glob("*/*/scores.json")):
        rows.append(json.loads(scores_file.read_text(encoding="utf-8")))

    if not rows:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    frame = pd.DataFrame(rows)
    frame["language"] = pd.Categorical(
        frame["language"],
        categories=LANGUAGE_ORDER,
        ordered=True,
    )
    frame = frame.sort_values(["run_id", "language"]).reset_index(drop=True)
    return frame


def _main_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in REPORT_COLUMNS if column in frame.columns]
    return frame.loc[:, columns].copy()


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _to_markdown(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format_cell(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _escape_latex(value: object) -> str:
    return (
        _format_cell(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _to_latex(frame: pd.DataFrame) -> str:
    column_spec = "l" * len(frame.columns)
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(_escape_latex(column) for column in frame.columns) + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(_escape_latex(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_reports(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = _main_matrix(frame)
    matrix.to_csv(output_dir / "test_main_matrix.csv", index=False)
    (output_dir / "test_main_matrix.md").write_text(
        _to_markdown(matrix),
        encoding="utf-8",
    )
    (output_dir / "test_main_matrix.tex").write_text(
        _to_latex(matrix),
        encoding="utf-8",
    )

    if "n_empty_predictions" in frame.columns:
        failures = matrix[frame["n_empty_predictions"] > 0]
    else:
        failures = matrix.iloc[0:0]
    failures.to_csv(output_dir / "test_main_matrix_failures.csv", index=False)
