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
    parser.add_argument(
        "dset", type=str, default="data",
        help="Dataset name")

    return parser.parse_args()


def generate_raw_files(prefix, frames, files, block_size, x_dim, y_dim,
                       dset="data"):

    values = []
    for _ in range(files):
        values.append(list())

    for frame_idx in range(frames):
        if files > 1:
            block_idx = frame_idx // block_size
            file_idx = block_idx % files
        else:
            file_idx = 0
        values[file_idx].append(frame_idx)

    for file_idx, file_values in enumerate(values):
        with h5py.File(prefix + "_{}.h5".format(file_idx)) as f:
            f.create_dataset(dset,
                             shape=(len(values[file_idx]),
                                    y_dim, x_dim))
            for idx, value in enumerate(file_values):
                f[dset][idx] = np.full((y_dim, x_dim),
                                        value, dtype="int8")


def main():
    """Run program."""
    args = parse_args()

    generate_raw_files(args.prefix, args.files, args.frames, args.block_size,
                       args.x_dim, args.y_dim, args.dset)


if __name__ == "__main__":
    sys.exit(main())
