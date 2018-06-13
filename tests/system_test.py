import os
import logging
from unittest import TestCase

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
        # Generate 4 raw files with interspersed frames
        # 95 2048x1536 frames, between 4 files in blocks of 10
        generate_raw_files("OD", 95, 4, 10, 2048, 1536)
        gen = InterleaveVDSGenerator(
            "./", prefix="OD_", block_size=10, log_level=1
        )
        gen.generate_vds()

    def test_sub_frames(self):
        # Generate 6 raw files each with 1/6th of a single 2048x1536 frame
        generate_raw_files("stripe", 6, 6, 1, 2048, 256)
        gen = SubFrameVDSGenerator(
            "./", prefix="stripe_", stripe_spacing=3, module_spacing=123,
            log_level=1
        )
        gen.generate_vds()

    def test_gap_fill(self):
        # Generate a single file with 100 2048x1536 frames
        generate_raw_files("raw", 100, 1, 1, 2048, 1536)
        gen = ExcaliburGapFillVDSGenerator(
            "./", files=["raw_0.h5"], chip_spacing=3, module_spacing=123,
            modules=3, output="gaps.h5", log_level=1
        )
        gen.generate_vds()
