import os
import sys
import unittest

from pkg_resources import require

require("mock")
from mock import MagicMock, patch, call

from vdsgen import vdsgenerator
from vdsgen.subframevdsgenerator import SubFrameVDSGenerator

vdsgen_patch_path = "vdsgen.subframevdsgenerator"
VDSGenerator_patch_path = vdsgen_patch_path + ".VDSGenerator"
h5py_patch_path = "h5py"

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "h5py"))


class SubFrameVDSGeneratorTester(SubFrameVDSGenerator):
    """A version of SubFrameVDSGenerator without initialisation.

    For testing single methods of the class. Must have required attributes
    passed before calling testee function.

    """

    def __init__(self, **kwargs):
        for attribute, value in kwargs.items():
            self.__setattr__(attribute, value)
        self.logger = MagicMock()


class FrameVDSGeneratorInitTest(unittest.TestCase):

    @patch(VDSGenerator_patch_path + '.__init__')
    def test_super_called(self, super_mock):
        SubFrameVDSGenerator("/test/path", prefix="stripe_")

        super_mock.assert_called_once_with("/test/path", "stripe_",
                                           *[None]*7)


class SimpleFunctionsTest(unittest.TestCase):

    @patch(VDSGenerator_patch_path + '.grab_metadata',
           return_value=dict(frames=(3,), height=256, width=2048,
                             dtype="uint16"))
    def test_process_source_datasets_given_valid_data(self, grab_mock):
        gen = SubFrameVDSGeneratorTester(
            datasets=["stripe_1.h5", "stripe_2.h5"])
        expected_source = vdsgenerator.SourceMeta(
            frames=(3,), height=256, width=2048, dtype="uint16")

        source = gen.process_source_datasets()

        grab_mock.assert_has_calls([call("stripe_1.h5"), call("stripe_2.h5")])
        self.assertEqual(expected_source, source)

    @patch(VDSGenerator_patch_path + '.grab_metadata',
           side_effect=[dict(frames=3, height=256, width=2048, dtype="uint16"),
                        dict(frames=4, height=256, width=2048,
                             dtype="uint16")])
    def test_process_source_datasets_given_mismatched_data(self, grab_mock):
        gen = SubFrameVDSGeneratorTester(
            datasets=["stripe_1.h5", "stripe_2.h5"])

        with self.assertRaises(ValueError):
            gen.process_source_datasets()

        grab_mock.assert_has_calls([call("stripe_1.h5"), call("stripe_2.h5")])

    def test_construct_vds_spacing(self):
        gen = SubFrameVDSGeneratorTester(
            datasets=[""] * 6, stripe_spacing=10, module_spacing=100)
        expected_spacing = [10, 100, 10, 100, 10, 0]

        spacing = gen.construct_vds_spacing()

        self.assertEqual(expected_spacing, spacing)

    @patch(vdsgen_patch_path + '.VirtualMap')
    @patch(vdsgen_patch_path + '.VirtualSource')
    @patch(vdsgen_patch_path + '.VirtualTarget')
    def test_create_vds_maps(self, target_mock, source_mock, map_mock):
        gen = SubFrameVDSGeneratorTester(
            output_file="/test/path/vds.hdf5",
            stripe_spacing=10, module_spacing=100,
            target_node="full_frame", source_node="data",
            datasets=["source"] * 6, name="vds.hdf5")
        source = vdsgenerator.SourceMeta(
            frames=(3,), height=256, width=2048, dtype="uint16")

        map_list = gen.create_vds_maps(source)

        target_mock.assert_called_once_with("/test/path/vds.hdf5",
                                            "full_frame",
                                            shape=(3, 1766, 2048))
        source_mock.assert_has_calls([call("source", "data",
                                           shape=(3, 256, 2048))] * 6)
        # TODO: Improve this assert by passing numpy arrays to check slicing
        map_mock.assert_has_calls([
            call(source_mock.return_value,
                 target_mock.return_value.__getitem__.return_value,
                 dtype="uint16")] * 6)
        self.assertEqual([map_mock.return_value] * 6, map_list)
