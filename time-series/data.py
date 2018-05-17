"""Sine-wave dataset generator.

   @author 
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola
  
   @project
     File: data.py
     Created on 17 May, 2018 @ 11:34 AM.
  
   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

import os
import argparse

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", type=int, default=100,
                        help="How many data points to be generated.")
    parser.add_argument("-seq", type=int, default=1000,
                        help="Sequence length")
    parser.add_argument("-t", "--step", type=int, default=20,
                        help="How many time steps to generate.")
    parser.add_argument("-s", "--save_dir", type=str, default="../datasets/time-series/sine-waves.pt",
                        help="File to save data to.")

    args = parser.parse_args()

    generate(args)


def generate(args):
    """Generate sine wave time series data.
    
    Args:
        args (): Command line arguments. 

    Returns:
        None
    """
    x = np.empty(shape=(args.n, args.seq), dtype=np.int64)

    seq = np.array(range(args.seq))
    rand = np.random.randint(low=-4 * args.step,
                             high=4 * args.step,
                             size=args.n)

    x[:] = seq + rand.reshape(args.n, 1)
    data = np.sin(x / 1.0 / args.step, dtype='float64')

    # Create save folders if `args.save_dir` contains directories.
    if len(args.save_dir.split('/')) > 1 and not os.path.isdir(os.path.dirname(args.save_dir)):
        os.makedirs(os.path.dirname(args.save_dir))

    # Save generated data.
    torch.save(data, open(args.save_dir, 'wb'))


if __name__ == '__main__':
    main()
