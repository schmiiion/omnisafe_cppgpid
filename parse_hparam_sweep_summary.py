#!/usr/bin/env python3
"""Parse the final-performance table from the CPPGPID hyperparameter sweep log."""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path


SUMMARY_MARKER = "Final-performance summary"
DEFAULT_OUTPUT_NAME = "hparam_sweep_cppgpid_final_performance.csv"
DEFAULT_SWEEP_DIR_NAMES = (
    "Hparam_Sweep_CPPGPID",
    "Hparam_sweep_CPPGPID",
    "Hparam_sweep_CPPPGPID",
)


def find_default_log_path() -> Path:
    """Find the sweep log next to this repository."""
    repo_dir = Path(__file__).resolve().parent
    workspace_dir = repo_dir.parent

    search_roots = (
        workspace_dir,
        workspace_dir / "processed_results",
    )

    for search_root in search_roots:
        for sweep_dir_name in DEFAULT_SWEEP_DIR_NAMES:
            log_path = search_root / sweep_dir_name / "log.txt"
            if log_path.exists():
                return log_path

    matches = sorted(workspace_dir.glob("**/Hparam*CPP*PID/log.txt"))
    if matches:
        return matches[0]

    return workspace_dir / DEFAULT_SWEEP_DIR_NAMES[0] / "log.txt"


def resolve_log_path(path: Path) -> Path:
    """Resolve a user-provided path to a concrete log file."""
    if path.is_dir():
        log_path = path / "log.txt"
        if log_path.exists():
            return log_path

        text_files = sorted(path.glob("*.txt"))
        if len(text_files) == 1:
            return text_files[0]

        raise FileNotFoundError(
            f"Could not choose a log file in {path}. Pass the log.txt path explicitly.",
        )

    return path


def split_markdown_row(line: str) -> list[str]:
    """Split a Markdown table row into stripped cells."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_alpha_cost(alpha_cost_expression: str) -> str:
    """Extract the value from an expression like alpha_cost=0.0."""
    key, separator, value = alpha_cost_expression.partition("=")
    if separator != "=" or key != "alpha_cost":
        raise ValueError(f"Expected an alpha_cost expression, got {alpha_cost_expression!r}")
    return value


def parse_final_performance_table(log_text: str) -> tuple[list[str], list[dict[str, str]]]:
    """Parse the final-performance table into CSV columns and rows."""
    lines = log_text.splitlines()

    try:
        marker_index = next(index for index, line in enumerate(lines) if SUMMARY_MARKER in line)
    except StopIteration as error:
        raise ValueError(f"Could not find the {SUMMARY_MARKER!r} section.") from error

    table_lines: list[str] = []
    inside_table = False
    for line in lines[marker_index + 1 :]:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)
            inside_table = True
        elif inside_table:
            break

    if len(table_lines) < 3:
        raise ValueError("The final-performance section does not contain a complete table.")

    header = split_markdown_row(table_lines[0])
    metric_columns = header[1:]
    columns = ["Env", "Algorithm", "alpha_cost", *metric_columns]

    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        cells = split_markdown_row(line)
        if len(cells) != len(header):
            raise ValueError(f"Unexpected table row shape: {line!r}")

        group_key = ast.literal_eval(cells[0])
        if not isinstance(group_key, tuple) or len(group_key) < 3:
            raise ValueError(f"Expected a tuple key with Env, Algorithm, alpha_cost: {cells[0]!r}")

        row = {
            "Env": str(group_key[0]),
            "Algorithm": str(group_key[1]),
            "alpha_cost": parse_alpha_cost(str(group_key[2])),
        }
        row.update(dict(zip(metric_columns, cells[1:])))
        rows.append(row)

    return columns, rows


def write_csv(columns: list[str], rows: list[dict[str, str]], output_path: Path) -> None:
    """Write parsed rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    repo_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Extract the CPPGPID final-performance summary table into CSV.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=find_default_log_path(),
        help="Path to the sweep log.txt or the sweep directory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=repo_dir / DEFAULT_OUTPUT_NAME,
        help=f"CSV output path. Defaults to {DEFAULT_OUTPUT_NAME} in this repository.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the parser."""
    args = parse_args()
    log_path = resolve_log_path(args.input)

    if args.output.is_dir():
        output_path = args.output / DEFAULT_OUTPUT_NAME
    else:
        output_path = args.output

    log_text = log_path.read_text(encoding="utf-8")
    columns, rows = parse_final_performance_table(log_text)
    if not rows:
        raise ValueError("No final-performance rows were parsed.")

    write_csv(columns, rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
