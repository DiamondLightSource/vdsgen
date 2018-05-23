import unittest
from pkg_resources import require

require("mock")
from mock import MagicMock, patch, call

from vdsgen import app

app_patch_path = "vdsgen.app"
parser_patch_path = app_patch_path + ".ArgumentParser"
VDSGenerator_patch_path = app_patch_path + ".VDSGenerator"
SubFrameVDSGenerator_patch_path = app_patch_path + ".SubFrameVDSGenerator"


class ParseArgsTest(unittest.TestCase):

    @patch(parser_patch_path + '.error')
    @patch(parser_patch_path + '.parse_args',
           return_value=MagicMock(empty=True, files=None))
    def test_empty_and_not_files_then_error(self, parse_mock, error_mock):

        app.parse_args()

        error_mock.assert_called_once_with(
            "To make an empty VDS you must explicitly define --files for the "
            "eventual raw datasets.")

    @patch(parser_patch_path + '.error')
    @patch(parser_patch_path + '.parse_args',
           return_value=MagicMock(empty=True, files=["file"]))
    def test_only_one_file_then_error(self, parse_mock, error_mock):

        app.parse_args()

        error_mock.assert_called_once_with(
            "Must define at least two files to combine.")


class MainTest(unittest.TestCase):

    @patch(SubFrameVDSGenerator_patch_path)
    @patch(app_patch_path + '.parse_args',
           return_value=MagicMock(
               path="/test/path", empty=True,
               prefix=None, files=["file1.hdf5", "file2.hdf5"], output="vds",
               shape=(3, 256, 2048), data_type="int16",
               source_node="data", target_node="full_frame",
               stripe_spacing=3, module_spacing=127, fill_value=-1,
               mode="sub-frames",
               log_level=2))
    def test_main_empty(self, parse_mock, init_mock):
        gen_mock = init_mock.return_value
        args_mock = parse_mock.return_value

        app.main()

        parse_mock.assert_called_once_with()
        init_mock.assert_called_once_with(
            args_mock.path,
            prefix=args_mock.prefix, files=args_mock.files,
            output=args_mock.output,
            source=dict(shape=args_mock.shape, dtype=args_mock.data_type),
            source_node=args_mock.source_node,
            target_node=args_mock.target_node,
            stripe_spacing=args_mock.stripe_spacing,
            module_spacing=args_mock.module_spacing,
            fill_value=-1,
            log_level=args_mock.log_level)

        gen_mock.generate_vds.assert_called_once_with()

    @patch("os.path.isfile", return_value=True)
    @patch(SubFrameVDSGenerator_patch_path)
    @patch(app_patch_path + '.parse_args',
           return_value=MagicMock(
               path="/test/path", empty=False,
               prefix=None, files=["file1.hdf5", "file2.hdf5"], output="vds",
               frames=3, height=256, width=2048, data_type="int16",
               source_node="data", target_node="full_frame",
               stripe_spacing=3, module_spacing=127, fill_value=-1,
               mode="sub-frames",
               log_level=2))
    def test_main_not_empty(self, parse_mock, generate_mock, _):
        args_mock = parse_mock.return_value

        app.main()

        parse_mock.assert_called_once_with()
        generate_mock.assert_called_once_with(
            args_mock.path,
            prefix=args_mock.prefix, output="vds", files=args_mock.files,
            source=None,
            source_node=args_mock.source_node,
            stripe_spacing=args_mock.stripe_spacing,
            target_node=args_mock.target_node,
            module_spacing=args_mock.module_spacing,
            fill_value=-1,
            log_level=args_mock.log_level)
