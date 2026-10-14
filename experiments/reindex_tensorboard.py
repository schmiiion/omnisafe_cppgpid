"""Write an existing OmniSafe progress.csv to TensorBoard using environment steps."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('progress_csv', type=Path)
    parser.add_argument(
        '--output-dir', type=Path,
        help='TensorBoard directory (default: tb_env_steps beside progress.csv)',
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.progress_csv.parent / 'tb_env_steps'
    if output_dir.exists():
        parser.error(f'Output directory already exists: {output_dir}')

    with args.progress_csv.open(newline='', encoding='utf-8') as source:
        rows = csv.DictReader(source)
        if not rows.fieldnames or 'TotalEnvSteps' not in rows.fieldnames:
            parser.error('progress.csv has no TotalEnvSteps column')
        with SummaryWriter(log_dir=str(output_dir)) as writer:
            for row in rows:
                step = int(float(row['TotalEnvSteps']))
                for key, raw_value in row.items():
                    if raw_value is None or raw_value == '':
                        continue
                    value = float(raw_value)
                    if math.isfinite(value):
                        writer.add_scalar(key, value, global_step=step)
    print(f'Wrote environment-step TensorBoard events to {output_dir}')


if __name__ == '__main__':
    main()
