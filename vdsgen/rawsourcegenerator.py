#!/bin/env dls-python
from pkg_resources import require
require("numpy")
require("h5py")

import numpy as np
import h5py

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "prefix", type=str, help="Path and name prefix")
    parser.add_argument(
        "frames", type=int, help="Number of frames.")
    parser.add_argument(
        "files", type=int, help="Number of files to spread frames across.")
    parser.add_argument(
        "block_size", type=int, default=1,
        help="Size of contiguous blocks of frames.")
    parser.add_argument(
        "x_dim", type=int, default=100,
        help="Width of frame")
    parser.add_argument(
        "y_dim", type=int, default=100,
        help="Height of frame")

    return parser.parse_args()


def main():
    """Run program."""
    args = parse_args()

    values = []
    for _ in range(args.files):
        values.append(list())

    for frame_idx in range(args.frames):
        if args.files > 1:
            block_idx = (frame_idx // (args.files - 1))
            file_idx = block_idx % args.files
        else:
            file_idx = 0
        values[file_idx].append(frame_idx + 1)

    for file_idx, file_values in enumerate(values):
        blocks = []
        for value in file_values:
            blocks.append(np.full((args.y_dim, args.x_dim), value))

        with h5py.File(args.prefix + "_{}.h5".format(file_idx)) as f:
            f.create_dataset("data", data=np.stack(blocks))


if __name__ == "__main__":
    sys.exit(main())
