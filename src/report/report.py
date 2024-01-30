import argparse
import json
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import classification_report


# testing output file schema:
# {
#     "true_labels": List[int],
#     "pred_labels": List[int],
#     "pred_prob": List[float]

# }


def generate_report(input_file: Path, output_file: Path):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    y_true: List[int] = data['true_labels']
    y_pred = data['pred_labels']
    report: Dict = classification_report(y_true, y_pred, output_dict=True)

    with open(output_file, 'w') as f:
        json.dump(report, f)


def main():
    parser = argparse.ArgumentParser(description='Report generator')

    parser.add_argument('input', type=str, required = True, help='Input file')
    parser.add_argument('output', type=str, required = True, help='Output file')

    args = parser.parse_args()

    input_file: Path = Path(args.input)
    output_file: Path = Path(args.output)

    generate_report(input_file, output_file)


if __name__ == '__main__':
    main()
