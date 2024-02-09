import argparse
import re
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def plot_loss_curve(epochs: List[int], train_losses: List[float], eval_losses: List[float], output_file: Path):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, eval_losses, label='Evaluation Loss', marker='o')
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid()
    plt.savefig(output_file)

def generate_log_summary_from_file(input_file: Path, output_file: Path):
    # Extract epoch and total loss
    epoch_pattern = r"Epoch (\d+)"
    loss_pattern = r"average loss: (\d+\.\d+)"
    epochs, train_losses, eval_losses = [], [], []
    for line in input_file.open('r'):
        if 'Epoch' in line:
            epoch = re.search(epoch_pattern, line).group(1)
            epochs.append(int(epoch))
        elif 'This epoch training' in line:
            loss = re.search(loss_pattern, line).group(1)
            train_losses.append(float(loss))
        elif 'This epoch eval' in line:
            loss = re.search(loss_pattern, line).group(1)
            eval_losses.append(float(loss))

    # truncate to the same length
    length = min(len(epochs), len(train_losses), len(eval_losses))
    epochs = epochs[:length]
    train_losses = train_losses[:length]
    eval_losses = eval_losses[:length]

    # plot loss curve
    plot_loss_curve(epochs, train_losses, eval_losses, output_file)


def main():
    parser = argparse.ArgumentParser(description='Report generator')

    parser.add_argument('--input', type=str, required=True, help='Input file')
    parser.add_argument('--output', type=str, required=True, help='Output file')

    args = parser.parse_args()

    input_file: Path = Path(args.input)
    output_file: Path = Path(args.output)

    generate_log_summary_from_file(input_file, output_file)


if __name__ == '__main__':
    main()
