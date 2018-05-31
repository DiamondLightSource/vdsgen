#!/bin/env dls-python
"""A class for generating virtual dataset frames from sub-frames."""

from .vdsgenerator import VDSGenerator, SourceMeta
from .group import VirtualSource, VirtualTarget, VirtualMap
from .hyperslab import Hyperslab


class GapFillVDSGenerator(VDSGenerator):

    """A class to generate a Virtual Dataset with gaps added to the source."""

    def __init__(self, path, prefix=None, files=None, output=None, source=None,
                 source_node=None, target_node=None, fill_value=None,
                 sub_width=256, sub_height=256, grid_x=8, grid_y=2,
                 log_level=None):
        """
        Args:
            path(str): Root folder to find raw files and create VDS
            prefix(str): Prefix of HDF5 files to generate from
                e.g. image_ for image_1.hdf5, image_2.hdf5, image_3.hdf5
            files(list(str)): List of HDF5 files to generate from
            output(str): Name of VDS file.
            source(dict): Height, width, data_type and frames for source data
                Provide this to create a VDS for raw files that don't exist yet
            source_node(str): Data node in source HDF5 files
            target_node(str): Data node in VDS file
            fill_value(int): Fill value for spacing
            sub_width(int): Width of sub-section in pixels
            sub_height(int): Height of sub-section in pixels
            grid_x(int): Width of full sensor in sub-sections
            grid_y(int): Height of full sensor in sub-sections
            log_level(int): Logging level (off=3, info=2, debug=1) -
                Default is info

        """
        self.sub_width = sub_width
        self.sub_height = sub_height
        self.grid_x = grid_x
        self.grid_y = grid_y

        super(GapFillVDSGenerator, self).__init__(
            path, prefix, files, output, source, source_node, target_node,
            fill_value, log_level)

        self.dataset = self.datasets[0]  # We only have one dataset

    def process_source_datasets(self):
        """Grab data from the given HDF5 files and check for consistency.

        Returns:
            Source: Number of datasets and the attributes of them (frames,
                height width and data type)

        """
        data = self.grab_metadata(self.datasets[0])

        source = SourceMeta(frames=data['frames'],
                            height=data['height'], width=data['width'],
                            dtype=data['dtype'])

        self.logger.debug("Source metadata retrieved:\n"
                          "  Frames: %s\n"
                          "  Height: %s\n"
                          "  Width: %s\n"
                          "  Data Type: %s", *source)
        return source

    def construct_vds_spacing(self):
        """Construct list of spacings between each sub-section.

        Returns:
            tuple(list): A list of spacings for horizontal and vertical gaps

        """
        raise NotImplementedError("Must be implemented in child class")

    def create_vds_maps(self, source_meta):
        """Create a list of VirtualMaps of raw data to the VDS.

        Args:
            source_meta(SourceMeta): Source attributes

        Returns:
            list(VirtualMap): Maps describing links between raw data and VDS

        """
        x_spacing, y_spacing = self.construct_vds_spacing()

        source_shape = source_meta.frames + (source_meta.height,
                                             source_meta.width)
        target_shape = source_meta.frames + \
                       (source_meta.height + sum(y_spacing),
                        source_meta.width + sum(x_spacing))
        self.logger.debug("VDS metadata:\n"
                          "  Shape: %s\n", target_shape)

        vds = VirtualTarget(self.output_file, self.target_node,
                            shape=target_shape)
        source = VirtualSource(self.dataset, self.source_node,
                               shape=source_shape)

        map_list = []
        y_current = 0
        for row in range(self.grid_y):
            y_start = y_current
            y_stop = y_start + self.sub_height + y_spacing[row]
            y_current = y_stop

            x_current = 0
            for section in range(self.grid_x):
                x_start = x_current
                x_stop = x_start + self.sub_width + x_spacing[section]
                x_current = x_stop

                # Hyperslab: All frames,
                #            Height bounds of chip,
                #            Width bounds of chip
                source_hyperslab = Hyperslab(
                    self.FULL_SLICE,
                    slice(self.sub_height * row,
                          self.sub_height * (row + 1)),
                    slice(self.sub_width * section,
                          self.sub_width * (section + 1))
                )

                # Hyperslab: All frames,
                #            Height bounds of chip with cumulative gap offset,
                #            Width bounds of chip with cumulative gap offset
                target_hyperslab = Hyperslab(
                    self.FULL_SLICE,
                    slice(y_start, y_stop),
                    slice(x_start, x_stop)
                )

                v_source = source[source_hyperslab.tuple]
                v_target = vds[target_hyperslab.tuple]
                v_map = VirtualMap(v_source, v_target, dtype=source_meta.dtype)

                self.logger.debug("Mapping dataset %s %s to %s of %s.",
                                  self.dataset.split("/")[-1],
                                  source_hyperslab,
                                  target_hyperslab, self.name)
                map_list.append(v_map)

        return map_list
