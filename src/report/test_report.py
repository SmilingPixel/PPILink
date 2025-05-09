import argparse
import json
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import classification_report, confusion_matrix


# testing output file schema:
# {
#     "true_labels": List[int],
#     "pred_labels": List[int],
#     "pred_prob": List[float]

# }


def generate_test_report(y_true: List[int], y_pred: List[int]) -> Dict:
    report: Dict = classification_report(y_true, y_pred, output_dict=True)
    report['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist() # convert to list for json serialization
    return report


def generate_test_report_file2file(input_file: Path, output_file: Path):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    y_true: List[int] = data['true_labels']
    y_pred = data['pred_labels']
    report: Dict = generate_test_report(y_true, y_pred)

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Report generator')

    parser.add_argument('--input', type=str, required=True, help='Input file')
    parser.add_argument('--output', type=str, required=True, help='Output file')

    args = parser.parse_args()

    input_file: Path = Path(args.input)
    output_file: Path = Path(args.output)

    generate_test_report_file2file(input_file, output_file)


if __name__ == '__main__':
    main()
