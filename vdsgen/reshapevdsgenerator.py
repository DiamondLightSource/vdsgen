"""A class to generate an ND Virtual Dataset from a 1D raw dataset."""

import logging

import h5py as h5

from vdsgenerator import VDSGenerator, SourceMeta


class ReshapeVDSGenerator(VDSGenerator):
    """A class to generate an ND Virtual Dataset from a 1D raw dataset."""

    logger = logging.getLogger("ReshapeVDSGenerator")

    def __init__(self, shape,
                 path, prefix=None, files=None, output=None, source=None,
                 source_node=None, target_node=None, fill_value=None,
                 log_level=None,
                 alternate=None):
        """
        Args:
            shape(tuple(int)): Shape of output dataset
            alternate(tuple(bool)): Whether each axis alternates

        """
        super(ReshapeVDSGenerator, self).__init__(
            path, prefix, files, output, source, source_node, target_node,
            fill_value, log_level)

        self.total_frames = 0
        self.periods = []
        self.alternate = alternate
        self.dimensions = shape
        self.source_file = self.files[0]  # Reshape only has one raw file

    def process_source_datasets(self):
        """Grab data from the given HDF5 files and check for consistency.

        Returns:
            Source: Number of datasets and the attributes of them (frames,
                height width and data type)

        """
        data = self.grab_metadata(self.files[0])
        self.total_frames = data["frames"][0]
        for dataset in self.files[1:]:
            temp_data = self.grab_metadata(dataset)
            self.total_frames += temp_data["frames"][0]
            for attribute, value in data.items():
                if attribute != "frames" and temp_data[attribute] != value:
                    raise ValueError("Files have mismatched "
                                     "{}".format(attribute))

        source = SourceMeta(frames=data['frames'],
                            height=data['height'], width=data['width'],
                            dtype=data['dtype'])

        self.logger.debug("Source metadata retrieved:\n"
                          "  Frames: %s\n"
                          "  Height: %s\n"
                          "  Width: %s\n"
                          "  Data Type: %s", self.total_frames, *source[1:])
        return source

    def create_virtual_layout(self, source_meta):
        """Create a VirtualLayout mapping raw data to the VDS.

        Args:
            source_meta(SourceMeta): Source attributes

        Returns:
            VirtualLayout: Object describing links between raw data and VDS

        """
        vds_shape = self.dimensions + (source_meta.height, source_meta.width)
        self.logger.debug("VDS metadata:\n"
                          "  Shape: %s\n", vds_shape)
        v_layout = h5.VirtualLayout(vds_shape, source_meta.dtype)

        source_shape = source_meta.frames + \
            (source_meta.height, source_meta.width)
        v_source = h5.VirtualSource(
            self.source_file,
            name="data", shape=source_shape, dtype=source_meta.dtype
        )

        v_layout[...] = v_source

        return v_layout
