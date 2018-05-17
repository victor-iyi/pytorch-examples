"""Train a 2-layer LSTM network to find patters in time-series data.

   @author 
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola
  
   @project
     File: train.py
     Created on 17 May, 2018 @ 11:57 AM.
  
   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
  
"""
import argparse

import numpy as np
import torch
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0,
                        help="Random number seed for deterministic output")
    parser.add_argument("--data_dir", type=str, default="../datasets/time-series/sine-waves.pt",
                        help="Directory to save learned model.")
    parser.add_argument("--save_dir", type=str, default="../saved/time-series/sine-waves.pt",
                        help="Directory to save learned model.")

    args = parser.parse_args()

    # Train the model.
    train(args)


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


class Sequence(nn.Module):
    def __init__(self, hidden_size=51):
        super(Sequence, self).__init__()

        # Hidden cell size.
        self.hidden_size = hidden_size

        # Network architecture.
        self.lstm1 = nn.LSTMCell(input_size=1,
                                 hidden_size=hidden_size)
        self.lstm2 = nn.LSTMCell(input_size=hidden_size,
                                 hidden_size=hidden_size)
        self.linear = nn.Linear(in_features=hidden_size,
                                out_features=1)

    def forward(self, inputs, future=0):
        input_size = inputs.size()
        outputs = []

        # Hidden & cell state for 1st time step.
        h_t1 = torch.zeros(input_size, self.hidden_size, dtype=torch.double)
        c_t1 = torch.zeros(input_size, self.hidden_size, dtype=torch.double)

        # Hidden & cell state for 2nd time step.
        h_t2 = torch.zeros(input_size, self.hidden_size, dtype=torch.double)
        c_t2 = torch.zeros(input_size, self.hidden_size, dtype=torch.double)

        for t, input_t in enumerate(inputs):
            h_t1, c_t1 = self.lstm1(input_t, (h_t1, c_t1))
            h_t2, c_t2 = self.lstm1(h_t1, (h_t2, c_t2))

if __name__ == '__main__':
    main()
