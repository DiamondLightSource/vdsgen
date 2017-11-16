import os
import sys
import unittest

from pkg_resources import require

require("mock")
from mock import MagicMock, patch, call

from vdsgen import vdsgenerator
from vdsgen.framevdsgenerator import FrameVDSGenerator

vdsgen_patch_path = "vdsgen.framevdsgenerator"
VDSGenerator_patch_path = vdsgen_patch_path + ".VDSGenerator"
h5py_patch_path = "h5py"

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "h5py"))


class FrameVDSGeneratorTester(FrameVDSGenerator):
    """A version of VDSGenerator without initialisation.

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
        gen = FrameVDSGenerator("/test/path", prefix="stripe_")

        self.assertEqual(0, gen.total_frames)
        super_mock.assert_called_once_with("/test/path", "stripe_",
                                           *[None]*7)


class SimpleFunctionsTest(unittest.TestCase):

    @patch(VDSGenerator_patch_path + '.grab_metadata',
           return_value=dict(frames=(5,), height=256, width=2048,
                             dtype="uint16"))
    def test_process_source_datasets_given_valid_data(self, grab_mock):
        gen = FrameVDSGeneratorTester(datasets=["stripe_1.h5", "stripe_2.h5"])
        expected_source = vdsgenerator.SourceMeta(
            frames=(5,), height=256, width=2048, dtype="uint16")

        source = gen.process_source_datasets()

        grab_mock.assert_has_calls([call("stripe_1.h5"), call("stripe_2.h5")])
        self.assertEqual(expected_source, source)
        self.assertEqual(10, gen.total_frames)

    @patch(VDSGenerator_patch_path + '.grab_metadata',
           side_effect=[
               dict(frames=(3,), height=256, width=2048, dtype="uint16"),
               dict(frames=(3,), height=512, width=2048, dtype="uint16")])
    def test_process_source_datasets_given_mismatched_data(self, grab_mock):
        gen = FrameVDSGeneratorTester(datasets=["stripe_1.h5", "stripe_2.h5"])

        with self.assertRaises(ValueError):
            gen.process_source_datasets()

        grab_mock.assert_has_calls([call("stripe_1.h5"), call("stripe_2.h5")])

    @patch(h5py_patch_path + '.File')
    @patch(vdsgen_patch_path + '.VirtualMap')
    @patch(vdsgen_patch_path + '.VirtualSource')
    @patch(vdsgen_patch_path + '.VirtualTarget')
    def test_create_vds_maps(self, target_mock, source_mock, map_mock,
                             file_mock):
        gen = FrameVDSGeneratorTester(
            output_file="/test/path/vds.hdf5",
            stripe_spacing=10, module_spacing=100,
            target_node="full_frame", source_node="data",
            datasets=["source"] * 2, name="vds.hdf5", shape=(5, 1586, 2048),
            total_frames=10)
        source = vdsgenerator.SourceMeta(
            frames=(5,), height=256, width=2048, dtype="uint16")
        file_mock.return_value = dict(data=MagicMock(shape=(5, 256, 2048)))

        map_list = gen.create_vds_maps(source)

        target_mock.assert_called_once_with("/test/path/vds.hdf5",
                                            "full_frame",
                                            shape=(10, 256, 2048))
        source_mock.assert_has_calls(
            [call("source", "data", shape=(5, 256, 2048))] * 2,
            any_order=True)
        # TODO: Improve this assert by passing numpy arrays to check slicing
        map_mock.assert_has_calls([
            call(source_mock.return_value.__getitem__.return_value,
                 target_mock.return_value.__getitem__.return_value,
                 dtype="uint16")] * 10,
                 any_order=True)
        self.assertEqual([map_mock.return_value] * 10, map_list)
