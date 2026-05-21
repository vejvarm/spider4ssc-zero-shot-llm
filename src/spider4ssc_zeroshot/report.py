from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

LANGUAGE_ORDER = ["sparql", "sql", "cypher"]
LANGUAGE_PALETTE = {
    "sql": "#B3A369",
    "sparql": "#367040",
    "cypher": "#5C7596",
}
REPORT_COLUMNS = [
    "run_id",
    "model_id",
    "split",
    "language",
    "schema_mode",
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


def _plot_width(n_models: int) -> float:
    return max(7.0, min(18.0, 2.0 + (1.4 * n_models)))


def _plot_grouped_bar(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    split: str,
    stem: str,
    value_column: str,
    ylabel: str,
    title: str,
) -> None:
    plot_data = frame.dropna(subset=["run_id", "language", value_column]).copy()
    if plot_data.empty:
        return

    run_order = sorted(str(run_id) for run_id in plot_data["run_id"].dropna().unique())
    plot_data["run_id"] = pd.Categorical(
        plot_data["run_id"].astype(str),
        categories=run_order,
        ordered=True,
    )
    plot_data["language"] = pd.Categorical(
        plot_data["language"].astype(str),
        categories=LANGUAGE_ORDER,
        ordered=True,
    )
    plot_data = plot_data.dropna(subset=["language"])
    if plot_data.empty:
        return

    fig, ax = plt.subplots(figsize=(_plot_width(len(run_order)), 4.8))
    sns.barplot(
        data=plot_data,
        x="run_id",
        y=value_column,
        hue="language",
        order=run_order,
        hue_order=LANGUAGE_ORDER,
        palette=LANGUAGE_PALETTE,
        errorbar=None,
        ax=ax,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.legend(title="Language", loc="upper right")
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(output_dir / f"{split}_{stem}{suffix}", dpi=200)
    plt.close(fig)


def _write_plots(frame: pd.DataFrame, output_dir: Path, *, split: str) -> None:
    if frame.empty:
        return

    plot_frame = frame.copy()
    plot_frame["execution_accuracy_percent"] = (
        pd.to_numeric(plot_frame["execution_accuracy"], errors="coerce") * 100
    )
    n_examples = pd.to_numeric(plot_frame["n_examples"], errors="coerce")
    n_empty = pd.to_numeric(plot_frame["n_empty_predictions"], errors="coerce")
    plot_frame["empty_prediction_rate_percent"] = (n_empty / n_examples) * 100

    _plot_grouped_bar(
        plot_frame,
        output_dir,
        split=split,
        stem="execution_accuracy_by_model",
        value_column="execution_accuracy_percent",
        ylabel="Execution accuracy (%)",
        title=f"{split.capitalize()} execution accuracy by model and language",
    )
    _plot_grouped_bar(
        plot_frame,
        output_dir,
        split=split,
        stem="empty_prediction_rate_by_model",
        value_column="empty_prediction_rate_percent",
        ylabel="Empty prediction rate (%)",
        title=f"{split.capitalize()} empty prediction rate by model and language",
    )


def write_reports(frame: pd.DataFrame, output_dir: Path, *, split: str = "test") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = _main_matrix(frame)
    matrix.to_csv(output_dir / f"{split}_main_matrix.csv", index=False)
    (output_dir / f"{split}_main_matrix.md").write_text(
        _to_markdown(matrix),
        encoding="utf-8",
    )
    (output_dir / f"{split}_main_matrix.tex").write_text(
        _to_latex(matrix),
        encoding="utf-8",
    )

    if "n_empty_predictions" in frame.columns:
        failures = matrix[matrix["n_empty_predictions"] > 0]
    else:
        failures = matrix.iloc[0:0]
    failures.to_csv(output_dir / f"{split}_main_matrix_failures.csv", index=False)
    _write_plots(matrix, output_dir, split=split)
