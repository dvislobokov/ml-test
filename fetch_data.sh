#!/bin/bash
# Скачивает MNIST и посимвольный корпус в data/
mkdir -p "$(dirname "$0")/data" && cd "$(dirname "$0")/data"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -sSfO https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz
done
curl -sSf -o tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
