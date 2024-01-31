import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


def generate_log_summary(input_file: Path, output_file: Path):
    #Extract epoch and total loss
    epoch_pattern = r"Epoch (\d+)"
    loss_pattern = r"total loss: (\d+\.\d+)"
    epochs, losses = [], []
    for line in input_file.open('r'):
        if 'Epoch' in line:
            epoch = re.search(epoch_pattern, line).group(1)
            epochs.append(int(epoch))
        if 'total loss' in line:
            loss = re.search(loss_pattern, line).group(1)
            losses.append(float(loss))

    # truncate to the same length
    length = min(len(epochs), len(losses))
    epochs = epochs[:length]
    losses = losses[:length]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o')
    plt.title('Total Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.grid()
    plt.savefig(output_file)


def main():
    parser = argparse.ArgumentParser(description='Report generator')

    parser.add_argument('--input', type=str, required=True, help='Input file')
    parser.add_argument('--output', type=str, required=True, help='Output file')

    args = parser.parse_args()

    input_file: Path = Path(args.input)
    output_file: Path = Path(args.output)

    generate_log_summary(input_file, output_file)


if __name__ == '__main__':
    main()
