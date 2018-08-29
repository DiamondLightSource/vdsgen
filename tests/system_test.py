from __future__ import print_function
import os
import sys
import logging
from unittest import TestCase

import h5py as h5

from vdsgen import SubFrameVDSGenerator, InterleaveVDSGenerator, \
    ExcaliburGapFillVDSGenerator, generate_raw_files


class SystemTest(TestCase):

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        for file_ in os.listdir("./"):
            if file_.endswith(".h5"):
                os.remove(file_)

    @classmethod
    def tearDownClass(cls):
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
        generate_raw_files("OD", FRAMES, 4, 10, WIDTH, HEIGHT)
        print("Creating VDS...")
        gen = InterleaveVDSGenerator(
            "./", prefix="OD_", block_size=10, log_level=1
        )
        gen.generate_vds()

        print("Opening VDS...")
        with h5.File("OD_vds.h5", mode="r") as h5_file:
            vds_dataset = h5_file["data"]

            print("Verifying dataset...")
            # Check shape
            self.assertEqual(vds_dataset.shape, (FRAMES, HEIGHT, WIDTH))
            # Check first pixel of each frame
            print("0 %", end="")
            for frame_idx in range(FRAMES):
                self.assertEqual(vds_dataset[frame_idx][0][0],
                                 1.0 * frame_idx)
                progress = int((frame_idx + 1) * (1.0 / FRAMES * 100))
                print("\r{} %".format(progress), end="")
                sys.stdout.flush()
            print()

    def test_sub_frames(self):
        FRAMES = 1
        WIDTH = 2048
        HEIGHT = 256
        FEMS = 6
        # Generate 6 raw files each with 1/6th of a single 2048x1536 frame
        print("Creating raw files...")
        generate_raw_files("stripe", FEMS * FRAMES, FEMS, 1, WIDTH, HEIGHT)
        print("Creating VDS...")
        gen = SubFrameVDSGenerator(
            "./", prefix="stripe_", stripe_spacing=3, module_spacing=123,
            log_level=1
        )
        gen.generate_vds()

        print("Opening VDS...")
        with h5.File("stripe_vds.h5", mode="r") as h5_file:
            vds_dataset = h5_file["data"]
            print("Verifying dataset...")
            # Check shape
            self.assertEqual(vds_dataset.shape, (FRAMES, 1791, WIDTH))
            # Check first pixel of each stripe and pixel above for fill value
            row = 0
            self.assertEqual(vds_dataset[0][row][0], 0.0)       # FEM 1 pixel 1
            row += 256 + 3                                      # Stripe space
            self.assertEqual(vds_dataset[0][row - 1][0], -1.0)  # Fill value
            self.assertEqual(vds_dataset[0][row][0], 1.0)       # FEM 2 pixel 1
            row += 256 + 123                                    # Module space
            self.assertEqual(vds_dataset[0][row - 1][0], -1.0)  # Fill value
            self.assertEqual(vds_dataset[0][row][0], 2.0)       # FEM 3 pixel 1
            row += 256 + 3                                      # Stripe space
            self.assertEqual(vds_dataset[0][row - 1][0], -1.0)  # Fill value
            self.assertEqual(vds_dataset[0][row][0], 3.0)       # FEM 4 pixel 1
            row += 256 + 123                                    # Module space
            self.assertEqual(vds_dataset[0][row - 1][0], -1.0)  # Fill value
            self.assertEqual(vds_dataset[0][row][0], 4.0)       # FEM 5 pixel 1
            row += 256 + 3                                      # Stripe space
            self.assertEqual(vds_dataset[0][row - 1][0], -1.0)  # Fill value
            self.assertEqual(vds_dataset[0][row][0], 5.0)       # FEM 6 pixel 1

    def test_gap_fill(self):
        # Generate a single file with 100 2048x1536 frames
        print("Creating raw files...")
        generate_raw_files("raw", 100, 1, 1, 2048, 1536)
        print("Creating VDS...")
        gen = ExcaliburGapFillVDSGenerator(
            "./", files=["raw_0.h5"], chip_spacing=3, module_spacing=123,
            modules=3, output="gaps.h5", log_level=1
        )
        gen.generate_vds()

        print("Opening VDS...")
        with h5.File("gaps.h5", mode="r") as h5_file:
            vds_dataset = h5_file["data"]
            print("Verifying dataset...")
            # Check shape
            self.assertEqual(vds_dataset.shape, (100, 1791, 2069))
            # Check first pixel of each stripe and pixel above for fill value
            row = 0
            self.assertEqual(vds_dataset[0][row][0], 0.0)       # FEM 1 pixel 1
            row += 256 + 3                                      # Stripe space
            self.assertEqual(vds_dataset[0][row - 1][0], -1.0)  # Fill value
            self.assertEqual(vds_dataset[0][row][0], 0.0)       # FEM 2 pixel 1
            row += 256 + 123                                    # Module space
            self.assertEqual(vds_dataset[0][row - 1][0], -1.0)  # Fill value
            self.assertEqual(vds_dataset[0][row][0], 0.0)       # FEM 3 pixel 1
            row += 256 + 3                                      # Stripe space
            self.assertEqual(vds_dataset[0][row - 1][0], -1.0)  # Fill value
            self.assertEqual(vds_dataset[0][row][0], 0.0)       # FEM 4 pixel 1
            row += 256 + 123                                    # Module space
            self.assertEqual(vds_dataset[0][row - 1][0], -1.0)  # Fill value
            self.assertEqual(vds_dataset[0][row][0], 0.0)       # FEM 5 pixel 1
            row += 256 + 3                                      # Stripe space
            self.assertEqual(vds_dataset[0][row - 1][0], -1.0)  # Fill value
            self.assertEqual(vds_dataset[0][row][0], 0.0)       # FEM 6 pixel 1
            # Check first pixel of each chip in first row
            col = 0
            self.assertEqual(vds_dataset[0][0][col], 0.0)       # Chip 1
            col += 256 + 3                                      # Chip space
            self.assertEqual(vds_dataset[0][0][col - 1], -1.0)  # Fill value
            self.assertEqual(vds_dataset[0][0][col], 0.0)       # Chip 2
            col += 256 + 3                                      # Chip space
            self.assertEqual(vds_dataset[0][0][col - 1], -1.0)  # Fill value
            self.assertEqual(vds_dataset[0][0][col], 0.0)       # Chip 3
            col += 256 + 3                                      # Chip space
            self.assertEqual(vds_dataset[0][0][col - 1], -1.0)  # Fill value
            self.assertEqual(vds_dataset[0][0][col], 0.0)       # Chip 4
            col += 256 + 3                                      # Chip space
            self.assertEqual(vds_dataset[0][0][col - 1], -1.0)  # Fill value
            self.assertEqual(vds_dataset[0][0][col], 0.0)       # Chip 5
            col += 256 + 3                                      # Chip space
            self.assertEqual(vds_dataset[0][0][col - 1], -1.0)  # Fill value
            self.assertEqual(vds_dataset[0][0][col], 0.0)       # Chip 6
