import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


def generate_log_summary(input_file: Path, output_file: Path):
    with open(input_file, 'r') as f:
        log = f.readlines()

    # Extract epoch and total loss
    pattern = r"Epoch (\d+).*total loss: (\d+\.\d+)"
    matches = re.findall(pattern, log)

    epochs = [int(epoch) for epoch, _ in matches]
    losses = [float(loss) for _, loss in matches]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o')
    plt.title('Total Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.grid()
    plt.savefig(output_file)


def main():
    parser = argparse.ArgumentParser(description='Report generator')

    parser.add_argument('input', type=str, required=True, help='Input file')
    parser.add_argument('output', type=str, required=True, help='Output file')

    args = parser.parse_args()

    input_file: Path = Path(args.input)
    output_file: Path = Path(args.output)

    generate_log_summary(input_file, output_file)


if __name__ == '__main__':
    main()
