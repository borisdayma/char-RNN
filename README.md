# Char-RNN

*This repository revisits text generation as detailed in [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and tries to bring more understanding about RNN's and how to optimize them.*

## Introduction

This repository contains support material to solve the text generation problem as detailed in [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). The objective is to read a large text file, one character at a time, and then be able to generate text (one character at a time) with the same style.

Every single experiment is automatically logged onto [Weighs & Biases](https://www.wandb.com/) for easier analysis/interpretation of results. We also want to bring more understanding about RNN's and how to optimize them.

## Usage

Dependencies can be installed through `requirements.txt`.

The following files are present:

- `notebook.ipynb` is a Jupyter Notebook used to prototype our solution ;
- `train.py` is a script to run several experiments and log them on [Weighs & Biases](https://www.wandb.com/).

## Results

See my results and conclusions:

- [W&B report](https://beta.wandb.ai/borisd13/char-RNN/reports?view=borisd13%2FReport)
- [W&B runs](https://beta.wandb.ai/borisd13/char-RNN?workspace=user-borisd13and)
