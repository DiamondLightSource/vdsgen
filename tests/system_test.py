from __future__ import print_function
import os
import sys
from timeit import default_timer as timer
import logging
from unittest import TestCase

import numpy as np
import h5py as h5

from vdsgen import SubFrameVDSGenerator, InterleaveVDSGenerator, \
    ExcaliburGapFillVDSGenerator, ReshapeVDSGenerator, generate_raw_files


def create_pixel_pattern(top_value, bottom_value,
                         horizontal_gap, vertical_gap,
                         fill_value=-1.0):
    WIDTH = 5
    pattern = [[top_value] * WIDTH,
               [bottom_value] * WIDTH]

    # Insert rows for horizontal gaps
    for _ in range(horizontal_gap):
        pattern.insert(1, [fill_value] * WIDTH)

    # Insert elements in all rows for vertical gaps
    if vertical_gap != 0:
        for row in pattern:
            row[1: WIDTH - 1] = [fill_value] * vertical_gap

    return np.array(pattern)


class SystemTest(TestCase):

    def setUp(self):
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        for file_ in os.listdir("./"):
            if file_.endswith(".h5"):
                os.remove(file_)

    def tearDown(self):
        for file_ in os.listdir("./"):
            if file_.endswith(".h5"):
                os.remove(file_)

    def test_interleave(self):
        FRAMES = 95
        WIDTH = 2048
        HEIGHT = 1536
        # Generate 4 raw files with interspersed frames
        # 95 2048x1536 frames, between 4 files in blocks of 10
        print("Creating raw files...")
        generate_raw_files("OD", FRAMES, 4, 10, WIDTH, HEIGHT, "test_data")
        print("Creating VDS...")
        gen = InterleaveVDSGenerator(
            "./", prefix="OD_", block_size=10, log_level=1,
            source_node="test_data"
        )
        gen.generate_vds()

        self._validate_interleave(FRAMES, HEIGHT, WIDTH)

    def test_interleave_empty(self):
        FRAMES = 95
        WIDTH = 2048
        HEIGHT = 1536
        print("Creating VDS...")
        gen = InterleaveVDSGenerator(
            "./", files=["OD_0.h5", "OD_1.h5", "OD_2.h5", "OD_3.h5"],
            source=dict(shape=((30, 25, 20, 20), HEIGHT, WIDTH),
                        dtype="float32"),
            block_size=10, log_level=1
        )
        gen.generate_vds()
        # Generate 4 raw files with interspersed frames
        # 95 2048x1536 frames, between 4 files in blocks of 10
        print("Creating raw files...")
        generate_raw_files("OD", FRAMES, 4, 10, WIDTH, HEIGHT)

        self._validate_interleave(FRAMES, HEIGHT, WIDTH)

    def _validate_interleave(self, frames, height, width):
        print("Opening VDS...")
        with h5.File("OD_vds.h5", mode="r") as h5_file:
            vds_dataset = h5_file["data"]

            print("Verifying dataset...")
            # Check shape
            self.assertEqual(vds_dataset.shape, (frames, height, width))
            # Check first pixel of each frame
            print("0 %", end="")
            for frame_idx in range(frames):
                self.assertEqual(vds_dataset[frame_idx][0][0],
                                 1.0 * frame_idx)
                progress = int((frame_idx + 1) * (1.0 / frames * 100))
                print("\r{} %".format(progress), end="")
                sys.stdout.flush()
            print()

    def test_interleave_speed(self):
        FILES = 4
        FRAMES = 10000
        WIDTH = 2048
        HEIGHT = 1536
        print("Creating VDS...")
        start = timer()
        gen = InterleaveVDSGenerator(
            "./", files=["OD_{}.h5".format(idx) for idx in range(FILES)],
            source=dict(shape=((FRAMES // FILES,) * FILES, HEIGHT, WIDTH),
                        dtype="float32"),
            block_size=10, log_level=1
        )
        gen.generate_vds()
        end = timer()
        print("Interleave {} frames - Time: {}".format(FRAMES, end - start))

    def test_sub_frames(self):
        FRAMES = 1
        WIDTH = 2048
        HEIGHT = 256
        FEMS = 6
        CHIPS = 8
        # Generate 6 raw files each with 1/6th of a single 2048x1536 frame
        print("Creating raw files...")
        generate_raw_files("stripe", FEMS * FRAMES, FEMS, 1, WIDTH, HEIGHT)
        print("Creating VDS...")
        gen = SubFrameVDSGenerator(
            "./", prefix="stripe_", stripe_spacing=3, module_spacing=123,
            log_level=1
        )
        gen.generate_vds()

        self._validate_sub_frames(FRAMES, WIDTH, FEMS, CHIPS)

    def test_sub_frames_empty(self):
        FRAMES = 1
        WIDTH = 2048
        HEIGHT = 256
        FEMS = 6
        CHIPS = 8
        print("Creating VDS...")
        gen = SubFrameVDSGenerator(
            "./", files=["stripe_0.h5", "stripe_1.h5", "stripe_2.h5",
                         "stripe_3.h5", "stripe_4.h5", "stripe_5.h5"],
            source=dict(shape=(FRAMES, HEIGHT, WIDTH), dtype="float32"),
            stripe_spacing=3, module_spacing=123,
            log_level=1
        )
        gen.generate_vds()
        # Generate 6 raw files each with 1/6th of a single 2048x1536 frame
        print("Creating raw files...")
        generate_raw_files("stripe", FEMS * FRAMES, FEMS, 1, WIDTH, HEIGHT)

        self._validate_sub_frames(FRAMES, WIDTH, FEMS, CHIPS)

    def _validate_sub_frames(self, frames, width, fems, chips):
        print("Opening VDS...")
        with h5.File("stripe_vds.h5", mode="r") as h5_file:
            vds_dataset = h5_file["data"]
            print("Verifying dataset...")
            # Check shape
            self.assertEqual(vds_dataset.shape, (frames, 1791, width))

            self.assertEqual(vds_dataset[0][0][0], 0.0)  # FEM 1 pixel 1

            # Check corners where 4 chips meet have expected pattern
            row = 255
            column = 255
            spacings = [3, 123, 3, 123, 3]
            frame = vds_dataset[0]
            for stripe_idx in range(fems - 1):  # FEMS - 1 Vertical Gaps
                spacing = spacings[stripe_idx]
                pattern = create_pixel_pattern(
                    top_value=1.0 * stripe_idx,
                    bottom_value=1.0 * (stripe_idx + 1),
                    horizontal_gap=spacing, vertical_gap=0, fill_value=-1.0
                )
                for chip_idx in range(chips - 1):  # CHIPS - 1 Horizontal Gaps
                    test = frame[row: row + spacing + 2, column: column + 5]
                    np.testing.assert_array_equal(pattern, test)
                    column += 256 + 3
                column = 255
                row += 256 + spacing

    def test_gap_fill(self):
        FEMS = 6
        CHIPS = 8
        FRAMES = 100
        HEIGHT = 1536
        WIDTH = 2048
        # Generate a single file with 100 2048x1536 frames
        print("Creating raw files...")
        generate_raw_files("raw", FRAMES, 1, 1, WIDTH, HEIGHT)
        print("Creating VDS...")
        gen = ExcaliburGapFillVDSGenerator(
            "./", files=["raw_0.h5"], chip_spacing=3, module_spacing=123,
            modules=3, output="gaps.h5", log_level=1
        )
        gen.generate_vds()

        self._validate_gap_fill(FEMS, CHIPS)

    def test_gap_fill_empty(self):
        FEMS = 6
        CHIPS = 8
        FRAMES = 100
        HEIGHT = 1536
        WIDTH = 2048
        print("Creating VDS...")
        gen = ExcaliburGapFillVDSGenerator(
            "./", files=["raw_0.h5"],
            source=dict(shape=(FRAMES, HEIGHT, WIDTH), dtype="float32"),
            chip_spacing=3, module_spacing=123,
            modules=3, output="gaps.h5", log_level=1
        )
        gen.generate_vds()
        # Generate a single file with 100 2048x1536 frames
        print("Creating raw files...")
        generate_raw_files("raw", FRAMES, 1, 1, WIDTH, HEIGHT)

        self._validate_gap_fill(FEMS, CHIPS)

    def _validate_gap_fill(self, fems, chips):
        print("Opening VDS...")
        with h5.File("gaps.h5", mode="r") as h5_file:
            vds_dataset = h5_file["data"]
            print("Verifying dataset...")
            # Check shape
            self.assertEqual(vds_dataset.shape, (100, 1791, 2069))

            self.assertEqual(vds_dataset[0][0][0], 0.0)  # FEM 1 pixel 1

            # Check corners where 4 chips meet have expected pattern
            row = 255
            column = 255
            spacings = [3, 123, 3, 123, 3]
            frame = vds_dataset[0]
            for stripe_idx in range(fems - 1):  # FEMS - 1 Vertical Gaps
                spacing = spacings[stripe_idx]
                pattern = create_pixel_pattern(
                    top_value=0.0, bottom_value=0.0,
                    horizontal_gap=spacing, vertical_gap=3, fill_value=-1.0
                )
                for chip_idx in range(chips - 1):  # CHIPS - 1 Horizontal Gaps
                    test = frame[row: row + spacing + 2, column: column + 5]
                    np.testing.assert_array_equal(pattern, test)
                    column += 256 + 3
                column = 255
                row += 256 + spacing

    def test_reshape(self):
        FRAMES = 60
        SHAPE = (5, 4, 3)
        # Generate a single file with 100 2048x1536 frames
        print("Creating raw files...")
        generate_raw_files("raw", FRAMES, 1, 1, 2048, 1536)
        print("Creating VDS...")
        gen = ReshapeVDSGenerator(
            shape=SHAPE, path="./", files=["raw_0.h5"],
            output="reshaped.h5", log_level=1
        )
        gen.generate_vds()

        self._validate_reshape(FRAMES)

    def test_reshape_alternate(self):
        FRAMES = 12
        SHAPE = (4, 3)
        # Generate a single file with 100 2048x1536 frames
        print("Creating raw files...")
        generate_raw_files("raw", FRAMES, 1, 1, 2048, 1536)
        print("Creating VDS...")
        gen = ReshapeVDSGenerator(
            shape=SHAPE, path="./", files=["raw_0.h5"],
            output="reshaped.h5", log_level=1,
            alternate=(False, True)
        )
        gen.generate_vds()

        test_dataset = np.array([0., 1., 2.,
                                 5., 4., 3.,
                                 6., 7., 8.,
                                 11., 10., 9.])

        with h5.File("reshaped.h5", mode="r") as h5_file:
            vds_dataset = h5_file["data"]

            print("Verifying dataset...")
            np.testing.assert_array_equal(
                test_dataset,
                vds_dataset[:, :, 0, 0].flatten()
            )

    def test_reshape_empty(self):
        FRAMES = 60
        SHAPE = (5, 4, 3)
        WIDTH = 2048
        HEIGHT = 1536
        print("Creating VDS...")
        gen = ReshapeVDSGenerator(
            shape=SHAPE, path="./", files=["raw_0.h5"],
            source=dict(shape=(FRAMES, HEIGHT, WIDTH), dtype="float32"),
            output="reshaped.h5", log_level=1
        )
        gen.generate_vds()
        # Generate a single file with 100 2048x1536 frames
        print("Creating raw files...")
        generate_raw_files("raw", FRAMES, 1, 1, 2048, 1536)

        self._validate_reshape(FRAMES)

    def test_reshape_speed(self):
        FRAMES = 1000000
        SHAPE = (500, 200, 10)
        WIDTH = 2048
        HEIGHT = 1536
        start = timer()
        print("Creating VDS...")
        gen = ReshapeVDSGenerator(
            shape=SHAPE, path="./", files=["raw_0.h5"],
            source=dict(shape=(FRAMES, HEIGHT, WIDTH), dtype="float32"),
            output="reshaped.h5", log_level=1
        )
        gen.generate_vds()
        end = timer()
        print("Reshape {} -> {} - Time: {}".format(FRAMES, SHAPE, end - start))

        start = timer()
        with h5.File("reshaped.h5", mode="r") as h5_file:
            dset = h5_file["data"]
        end = timer()
        print("Dataset open time: {}".format(end - start))

    def test_reshape_alternate_speed(self):
        FRAMES = 10000
        SHAPE = (50, 20, 10)
        WIDTH = 2048
        HEIGHT = 1536
        start = timer()
        print("Creating VDS...")
        gen = ReshapeVDSGenerator(
            shape=SHAPE, path="./", files=["raw_0.h5"],
            source=dict(shape=(FRAMES, HEIGHT, WIDTH), dtype="float32"),
            output="reshaped.h5", log_level=1, alternate=(False, True, True)
        )
        gen.generate_vds()
        end = timer()
        print("Reshape {} -> {} - Time: {}".format(FRAMES, SHAPE, end - start))

        start = timer()
        with h5.File("reshaped.h5", mode="r") as h5_file:
            dset = h5_file["data"]
        end = timer()
        print("Dataset open time: {}".format(end - start))

    def _validate_reshape(self, frames):
        print("Opening VDS...")
        with h5.File("reshaped.h5", mode="r") as h5_file:
            vds_dataset = h5_file["data"]

            print("Verifying dataset...")
            np.testing.assert_array_equal(
                np.array(range(frames)),
                vds_dataset[:, :, :, 0, 0].flatten()
            )
