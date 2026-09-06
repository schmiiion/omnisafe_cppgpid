#!/usr/bin/env python3
"""Plot or export final-performance tables for the CPPGPID hyperparameter sweep."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT_NAME = "hparam_sweep_cppgpid_final_performance.csv"
DEFAULT_RESULTS_DIR = "processed_results"
BASELINE_ALGORITHMS = {
    "CPPGPID": "CPPOPID",
    "PTRPOPID": "TRPOPID",
    "PhasicCPO": "CPO",
}
LATEX_ARRAY_STRETCH = "1.35"
LATEX_ROW_EXTRA_SPACE = "0.65em"
REQUIRED_COLUMNS = (
    "Env",
    "Algorithm",
    "alpha_cost",
    "Metrics/EpRet median",
    "Metrics/EpRet IQR",
    "Metrics/EpCost median",
    "Metrics/EpCost IQR",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    repo_dir = Path(__file__).resolve().parent
    default_out_all = repo_dir.parent / DEFAULT_RESULTS_DIR / "outAll"

    parser = argparse.ArgumentParser(
        description="Plot one final-performance table per environment.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=repo_dir / DEFAULT_INPUT_NAME,
        help=f"Path to the parsed CSV. Defaults to {DEFAULT_INPUT_NAME}.",
    )
    parser.add_argument(
        "--return-table",
        type=Path,
        default=default_out_all / "return_table.csv",
        help="Baseline return table CSV.",
    )
    parser.add_argument(
        "--cost-table",
        type=Path,
        default=default_out_all / "cost_table.csv",
        help="Baseline cost table CSV.",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not add the baseline column from return_table.csv and cost_table.csv.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print LaTeX tables to stdout instead of showing matplotlib figures.",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        help="Write LaTeX tables to this file. Combine with --no-show to skip figures.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plot.show(). Useful when only exporting LaTeX.",
    )
    return parser.parse_args()


def read_csv_rows(input_path: Path) -> list[dict[str, str]]:
    """Read a CSV file as string-valued rows."""
    with input_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []
        validate_columns(fieldnames)
        return [{key: value for key, value in row.items()} for row in reader]


def validate_columns(fieldnames: list[str]) -> None:
    """Validate that the CSV has the columns required by the plotter."""
    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        missing_columns = ", ".join(missing)
        raise ValueError(f"CSV is missing required columns: {missing_columns}")


def read_metric_table(input_path: Path) -> dict[str, dict[str, str]]:
    """Read one outAll metric table keyed by algorithm and environment."""
    with input_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames or []
        algorithm_column = "algorithm" if "algorithm" in fieldnames else "Algorithm"
        if algorithm_column not in fieldnames:
            raise ValueError(f"{input_path} must contain an algorithm column.")

        table: dict[str, dict[str, str]] = {}
        for row in reader:
            algorithm = row[algorithm_column]
            table[algorithm] = {
                env: value
                for env, value in row.items()
                if env != algorithm_column and value is not None
            }
        return table


def read_baseline_tables(
    return_table_path: Path,
    cost_table_path: Path,
    no_baseline: bool,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]] | None:
    """Read baseline return and cost tables, unless disabled."""
    if no_baseline:
        return None

    missing = [
        path for path in (return_table_path, cost_table_path) if not path.exists()
    ]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Baseline table file(s) not found: {missing_paths}. Use --no-baseline to skip them.",
        )

    return read_metric_table(return_table_path), read_metric_table(cost_table_path)


def unique_in_order(values: list[str]) -> list[str]:
    """Return unique values while preserving their first-seen order."""
    return list(dict.fromkeys(values))


def alpha_sort_key(value: str) -> tuple[int, float | str]:
    """Sort alpha values numerically when possible."""
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def metric_text(row: dict[str, str]) -> tuple[str, str]:
    """Build the EpRet and EpCost strings for one sweep table cell."""
    ret = f"{row['Metrics/EpRet median']} +/- {row['Metrics/EpRet IQR']}"
    cost = f"{row['Metrics/EpCost median']} +/- {row['Metrics/EpCost IQR']}"
    return ret, cost


def get_baseline_text(
    algorithm: str,
    env: str,
    baseline_tables: tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]] | None,
) -> tuple[str, str, str] | None:
    """Return baseline algorithm, return text, and cost text for one row."""
    if baseline_tables is None:
        return None

    baseline_algorithm = BASELINE_ALGORITHMS.get(algorithm)
    if baseline_algorithm is None:
        return None

    return_table, cost_table = baseline_tables
    ret = return_table.get(baseline_algorithm, {}).get(env)
    cost = cost_table.get(baseline_algorithm, {}).get(env)
    if ret is None or cost is None:
        return None

    return baseline_algorithm, ret, cost


def draw_cell(
    axis: Any,
    rectangle_cls: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
) -> None:
    """Draw one rectangular table cell."""
    axis.add_patch(
        rectangle_cls(
            (x, y),
            width,
            height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.8,
        ),
    )


def draw_split_cell_text(
    axis: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    left_text: str,
    right_text: str,
    left_color: str,
    right_color: str,
    fontsize: float,
    title: str | None = None,
) -> None:
    """Draw return text on the left and cost text on the right of a cell."""
    if title:
        axis.text(
            x + width / 2,
            y + height * 0.76,
            title,
            ha="center",
            va="center",
            fontsize=max(fontsize - 0.2, 5.0),
            fontweight="bold",
        )
        text_y = y + height * 0.35
    else:
        text_y = y + height / 2

    axis.text(
        x + width * 0.25,
        text_y,
        left_text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=left_color,
    )
    axis.text(
        x + width * 0.75,
        text_y,
        right_text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=right_color,
    )


def draw_environment_table(
    axis: Any,
    rectangle_cls: Any,
    rows: list[dict[str, str]],
    env: str,
    baseline_tables: tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]] | None,
) -> None:
    """Draw one table for a single environment."""
    env_rows = [row for row in rows if row["Env"] == env]
    algorithms = unique_in_order([row["Algorithm"] for row in env_rows])
    alpha_values = sorted(
        unique_in_order([row["alpha_cost"] for row in env_rows]),
        key=alpha_sort_key,
    )
    row_lookup = {
        (row["Algorithm"], row["alpha_cost"]): row
        for row in env_rows
    }

    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    grid_color = "#9aa3ad"
    header_color = "#e9edf2"
    row_header_color = "#f5f7fa"
    ret_color = "#183a59"
    cost_color = "#6f2f2f"

    left = 0.015
    bottom = 0.04
    table_width = 0.97
    table_height = 0.82
    header_height = 0.16
    algorithm_width = 0.13
    has_baseline = baseline_tables is not None
    value_column_count = len(alpha_values) + int(has_baseline)
    data_width = (table_width - algorithm_width) / value_column_count
    row_height = (table_height - header_height) / len(algorithms)
    table_top = bottom + table_height
    value_fontsize = 6.5 if has_baseline else 8.0
    baseline_fontsize = 5.6

    axis.text(
        0.5,
        0.95,
        env,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    draw_cell(
        axis,
        rectangle_cls,
        left,
        table_top - header_height,
        algorithm_width,
        header_height,
        header_color,
        grid_color,
    )
    axis.text(
        left + algorithm_width / 2,
        table_top - header_height / 2,
        "Algorithm",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    for alpha_index, alpha_cost in enumerate(alpha_values):
        x = left + algorithm_width + alpha_index * data_width
        y = table_top - header_height
        draw_cell(axis, rectangle_cls, x, y, data_width, header_height, header_color, grid_color)
        axis.plot(
            [x + data_width / 2, x + data_width / 2],
            [y, y + header_height],
            color=grid_color,
            linewidth=0.6,
        )
        axis.text(
            x + data_width / 2,
            y + header_height * 0.68,
            f"alpha_cost={alpha_cost}",
            ha="center",
            va="center",
            fontsize=8.4,
            fontweight="bold",
        )
        axis.text(
            x + data_width * 0.25,
            y + header_height * 0.28,
            "EpRet",
            ha="center",
            va="center",
            fontsize=8,
            color=ret_color,
            fontweight="bold",
        )
        axis.text(
            x + data_width * 0.75,
            y + header_height * 0.28,
            "EpCost",
            ha="center",
            va="center",
            fontsize=8,
            color=cost_color,
            fontweight="bold",
        )

    if has_baseline:
        baseline_x = left + algorithm_width + len(alpha_values) * data_width
        y = table_top - header_height
        draw_cell(
            axis,
            rectangle_cls,
            baseline_x,
            y,
            data_width,
            header_height,
            header_color,
            grid_color,
        )
        axis.plot(
            [baseline_x + data_width / 2, baseline_x + data_width / 2],
            [y, y + header_height],
            color=grid_color,
            linewidth=0.6,
        )
        axis.text(
            baseline_x + data_width / 2,
            y + header_height * 0.68,
            "Baseline",
            ha="center",
            va="center",
            fontsize=8.4,
            fontweight="bold",
        )
        axis.text(
            baseline_x + data_width * 0.25,
            y + header_height * 0.28,
            "EpRet",
            ha="center",
            va="center",
            fontsize=8,
            color=ret_color,
            fontweight="bold",
        )
        axis.text(
            baseline_x + data_width * 0.75,
            y + header_height * 0.28,
            "EpCost",
            ha="center",
            va="center",
            fontsize=8,
            color=cost_color,
            fontweight="bold",
        )

    for algorithm_index, algorithm in enumerate(algorithms):
        y = table_top - header_height - (algorithm_index + 1) * row_height

        draw_cell(axis, rectangle_cls, left, y, algorithm_width, row_height, row_header_color, grid_color)
        axis.text(
            left + algorithm_width / 2,
            y + row_height / 2,
            algorithm,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

        for alpha_index, alpha_cost in enumerate(alpha_values):
            x = left + algorithm_width + alpha_index * data_width
            draw_cell(axis, rectangle_cls, x, y, data_width, row_height, "white", grid_color)
            axis.plot(
                [x + data_width / 2, x + data_width / 2],
                [y, y + row_height],
                color=grid_color,
                linewidth=0.6,
            )

            row = row_lookup.get((algorithm, alpha_cost))
            if row is None:
                ret, cost = "-", "-"
            else:
                ret, cost = metric_text(row)

            draw_split_cell_text(
                axis,
                x,
                y,
                data_width,
                row_height,
                ret,
                cost,
                ret_color,
                cost_color,
                value_fontsize,
            )

        if has_baseline:
            baseline_x = left + algorithm_width + len(alpha_values) * data_width
            draw_cell(
                axis,
                rectangle_cls,
                baseline_x,
                y,
                data_width,
                row_height,
                "white",
                grid_color,
            )
            axis.plot(
                [baseline_x + data_width / 2, baseline_x + data_width / 2],
                [y, y + row_height],
                color=grid_color,
                linewidth=0.6,
            )

            baseline = get_baseline_text(algorithm, env, baseline_tables)
            if baseline is None:
                baseline_algorithm, ret, cost = "", "-", "-"
            else:
                baseline_algorithm, ret, cost = baseline

            draw_split_cell_text(
                axis,
                baseline_x,
                y,
                data_width,
                row_height,
                ret,
                cost,
                ret_color,
                cost_color,
                baseline_fontsize,
                baseline_algorithm,
            )


def plot_tables(
    rows: list[dict[str, str]],
    baseline_tables: tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]] | None,
) -> None:
    """Create one table figure per environment."""
    import matplotlib.pyplot as plot
    import seaborn as sns
    from matplotlib.patches import Rectangle

    sns.set_theme(style="white")

    envs = unique_in_order([row["Env"] for row in rows])
    alpha_count = max(
        len(unique_in_order([row["alpha_cost"] for row in rows if row["Env"] == env]))
        for env in envs
    )
    column_count = alpha_count + int(baseline_tables is not None)
    figure_width = max(16.0, 3.55 * column_count + 2.0)

    for env in envs:
        figure, axis = plot.subplots(figsize=(figure_width, 4.8))
        draw_environment_table(axis, Rectangle, rows, env, baseline_tables)
        figure.tight_layout()


def latex_escape(text: str) -> str:
    """Escape text for LaTeX."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def latex_label(text: str) -> str:
    """Create a simple LaTeX-safe label suffix."""
    label = []
    previous_dash = False
    for character in text.lower():
        if character.isalnum():
            label.append(character)
            previous_dash = False
        elif not previous_dash:
            label.append("-")
            previous_dash = True
    return "".join(label).strip("-")


def parse_leading_float(text: str) -> float | None:
    """Parse the leading numeric result from a metric string."""
    match = re.match(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", text)
    if match is None:
        return None
    return float(match.group(1))


def is_best(value: float | None, best_value: float | None) -> bool:
    """Return whether a value should be highlighted as best."""
    if value is None or best_value is None:
        return False
    return abs(value - best_value) <= 1e-12


def max_present(values: list[float | None]) -> float | None:
    """Return the maximum value from a list that may contain missing values."""
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    return max(present_values)


def min_present(values: list[float | None]) -> float | None:
    """Return the minimum value from a list that may contain missing values."""
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    return min(present_values)


def latex_maybe_bold(text: str, bold: bool) -> str:
    """Escape a string and optionally wrap it in LaTeX bold markup."""
    escaped = latex_escape(text)
    if bold:
        return r"\textbf{" + escaped + "}"
    return escaped


def latex_bold_leading_value(text: str, bold: bool) -> str:
    """Escape a metric string and optionally bold its leading numeric value."""
    match = re.match(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(.*)", text)
    if match is None:
        return latex_escape(text)

    value, rest = match.groups()
    return latex_maybe_bold(value, bold) + latex_escape(rest)


def latex_sweep_metric_text(
    row: dict[str, str],
    bold_ret: bool,
    bold_cost: bool,
) -> tuple[str, str]:
    """Build LaTeX metric strings for one sweep row."""
    ret = (
        f"{latex_maybe_bold(row['Metrics/EpRet median'], bold_ret)}"
        f" $\\pm$ {latex_escape(row['Metrics/EpRet IQR'])}"
    )
    cost = (
        f"{latex_maybe_bold(row['Metrics/EpCost median'], bold_cost)}"
        f" $\\pm$ {latex_escape(row['Metrics/EpCost IQR'])}"
    )
    return ret, cost


def latex_inner_cell(lines: list[str]) -> str:
    """Build a multi-line LaTeX cell without requiring makecell."""
    return r"\begin{tabular}{@{}c@{}}" + r" \\ ".join(lines) + r"\end{tabular}"


def build_latex_tables(
    rows: list[dict[str, str]],
    baseline_tables: tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]] | None,
) -> str:
    """Build pasteable LaTeX code with one table per environment."""
    tables: list[str] = []
    envs = unique_in_order([row["Env"] for row in rows])

    for env in envs:
        env_rows = [row for row in rows if row["Env"] == env]
        algorithms = unique_in_order([row["Algorithm"] for row in env_rows])
        alpha_values = sorted(
            unique_in_order([row["alpha_cost"] for row in env_rows]),
            key=alpha_sort_key,
        )
        row_lookup = {
            (row["Algorithm"], row["alpha_cost"]): row
            for row in env_rows
        }
        has_baseline = baseline_tables is not None
        column_count = len(alpha_values) + int(has_baseline)
        caption = (
            r"\caption{Final performance on "
            + latex_escape(env)
            + r". Sweep entries report median $\pm$ IQR"
        )
        if has_baseline:
            caption += (
                r"; baseline entries report the median with the reported interval.}"
            )
        else:
            caption += r".}"

        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\scriptsize",
            rf"\renewcommand{{\arraystretch}}{{{LATEX_ARRAY_STRETCH}}}",
            r"\setlength{\tabcolsep}{3pt}",
            caption,
            r"\label{tab:hparam-" + latex_label(env) + r"}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{l" + ("c" * column_count) + r"}",
            r"\hline",
        ]

        header = [
            "Algorithm",
            *[
                r"$\alpha_{\mathrm{cost}}="
                + latex_escape(alpha_cost)
                + r"$"
                for alpha_cost in alpha_values
            ],
        ]
        if has_baseline:
            header.append("Baseline")
        lines.append(" & ".join(header) + r" \\")
        lines.append(r"\hline")

        for algorithm_index, algorithm in enumerate(algorithms):
            baseline = get_baseline_text(algorithm, env, baseline_tables)

            ret_values: list[float | None] = []
            cost_values: list[float | None] = []
            for alpha_cost in alpha_values:
                row = row_lookup.get((algorithm, alpha_cost))
                if row is not None:
                    ret_values.append(parse_leading_float(row["Metrics/EpRet median"]))
                    cost_values.append(parse_leading_float(row["Metrics/EpCost median"]))
            if baseline is not None:
                ret_values.append(parse_leading_float(baseline[1]))
                cost_values.append(parse_leading_float(baseline[2]))

            best_ret = max_present(ret_values)
            best_cost = min_present(cost_values)

            cells = [latex_escape(algorithm)]
            for alpha_cost in alpha_values:
                row = row_lookup.get((algorithm, alpha_cost))
                if row is None:
                    cells.append("-")
                else:
                    ret_value = parse_leading_float(row["Metrics/EpRet median"])
                    cost_value = parse_leading_float(row["Metrics/EpCost median"])
                    ret, cost = latex_sweep_metric_text(
                        row,
                        is_best(ret_value, best_ret),
                        is_best(cost_value, best_cost),
                    )
                    cells.append(latex_inner_cell([f"Ret: {ret}", f"Cost: {cost}"]))

            if has_baseline:
                if baseline is None:
                    cells.append("-")
                else:
                    _, ret, cost = baseline
                    cells.append(
                        latex_inner_cell(
                            [
                                (
                                    "Ret: "
                                    + latex_bold_leading_value(
                                        ret,
                                        is_best(parse_leading_float(ret), best_ret),
                                    )
                                ),
                                (
                                    "Cost: "
                                    + latex_bold_leading_value(
                                        cost,
                                        is_best(parse_leading_float(cost), best_cost),
                                    )
                                ),
                            ],
                        ),
                    )

            lines.append(" & ".join(cells) + r" \\")
            if algorithm_index < len(algorithms) - 1:
                lines.append(rf"\noalign{{\vskip {LATEX_ROW_EXTRA_SPACE}}}")
                lines.append(r"\hline")
                lines.append(rf"\noalign{{\vskip {LATEX_ROW_EXTRA_SPACE}}}")

        lines.extend(
            [
                r"\hline",
                r"\end{tabular}%",
                r"}",
                r"\end{table*}",
            ],
        )
        tables.append("\n".join(lines))

    return "\n\n".join(tables) + "\n"


def main() -> None:
    """Read the CSV and either show plotted tables or export LaTeX."""
    args = parse_args()
    rows = read_csv_rows(args.input)
    baseline_tables = read_baseline_tables(
        args.return_table,
        args.cost_table,
        args.no_baseline,
    )

    if args.latex or args.latex_output:
        latex = build_latex_tables(rows, baseline_tables)
        if args.latex_output:
            args.latex_output.parent.mkdir(parents=True, exist_ok=True)
            args.latex_output.write_text(latex, encoding="utf-8")
        if args.latex:
            print(latex, end="")

    if not args.latex and not args.no_show:
        plot_tables(rows, baseline_tables)
        import matplotlib.pyplot as plot

        plot.show()


if __name__ == "__main__":
    main()
