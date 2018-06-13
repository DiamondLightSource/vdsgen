#!/bin/env dls-python
"""A class for generating virtual dataset frames from sub-frames."""

import h5py as h5

from .vdsgenerator import VDSGenerator, SourceMeta
from .group import VirtualSource, VirtualTarget, VirtualMap
from .hyperslab import Hyperslab


class InterleaveVDSGenerator(VDSGenerator):

    """A class to generate Virtual Dataset frames from sub-frames."""

    def __init__(self, path, prefix=None, files=None, output=None, source=None,
                 source_node=None, target_node=None, fill_value=None,
                 block_size=1,
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
            block_size(int): Number of contiguous frames per block
            log_level(int): Logging level (off=3, info=2, debug=1) -
                Default is info

        """
        self.block_size = block_size
        self.total_frames = 0

        super(InterleaveVDSGenerator, self).__init__(
            path, prefix, files, output, source, source_node, target_node,
            fill_value, log_level)

    def process_source_datasets(self):
        """Grab data from the given HDF5 files and check for consistency.

        Returns:
            Source: Number of datasets and the attributes of them (frames,
                height width and data type)

        """
        data = self.grab_metadata(self.datasets[0])
        self.total_frames = data["frames"][0]
        for dataset in self.datasets[1:]:
            temp_data = self.grab_metadata(dataset)
            self.total_frames += temp_data["frames"][0]
            for attribute, value in data.items():
                if attribute != "frames" and temp_data[attribute] != value:
                    raise ValueError("Files have mismatched "
                                     "{}".format(attribute))

        source = SourceMeta(frames=self.total_frames,
                            height=data['height'], width=data['width'],
                            dtype=data['dtype'])

        self.logger.debug("Source metadata retrieved:\n"
                          "  Frames: %s\n"
                          "  Height: %s\n"
                          "  Width: %s\n"
                          "  Data Type: %s", self.total_frames, *source[1:])
        return source

    def create_vds_maps(self, source_meta):
        """Create a list of VirtualMaps of raw data to the VDS.

        Args:
            source_meta(SourceMeta): Source attributes

        Returns:
            list(VirtualMap): Maps describing links between raw data and VDS

        """
        source_dims = (source_meta.height, source_meta.width)
        target_shape = (self.total_frames,
                        source_meta.height, source_meta.width)
        self.logger.debug("VDS metadata:\n"
                          "  Shape: %s\n", target_shape)

        vds = VirtualTarget(self.output_file, self.target_node,
                            shape=target_shape)

        total_datasets = len(self.datasets)

        map_list = []
        for dataset_idx, dataset in enumerate(self.datasets):
            dataset_frames = h5.File(dataset)[self.source_node].shape[0]
            source = VirtualSource(dataset, self.source_node,
                                   shape=(dataset_frames,)+source_dims)

            for frame_idx in range(0, dataset_frames, self.block_size):

                source_block_idx = frame_idx // self.block_size
                target_block_idx = \
                    dataset_idx + total_datasets * source_block_idx

                # Make sure to only take as many frames as raw dataset has
                # It is OK if this isn't a complete block if it is the last one
                block_size = min(self.block_size, dataset_frames - frame_idx)

                if block_size < self.block_size:
                    self.logger.warning(
                        "%s does not have integer number of blocks sized %s",
                        dataset.split("/")[-1], self.block_size)

                # Hyperslab: Frame[block_start_idx, block_end_idx + 1],
                #            Full slice for height and width
                source_hyperslab = Hyperslab(
                    slice(frame_idx, frame_idx + block_size),
                    self.FULL_SLICE, self.FULL_SLICE)
                v_source = source[source_hyperslab.tuple]

                block_start = self.block_size * target_block_idx
                if block_start + block_size > self.total_frames:
                    raise RuntimeError(
                        "Cannot map dataset %d of %d [%d frames] "
                        "to vds [%d frames] with blocks sized %d" %
                        (self.datasets.index(dataset) + 1, len(self.datasets),
                         dataset_frames, self.total_frames, self.block_size))

                # Hyperslab: Frame[block_start_idx, block_end_idx + 1],
                #            Full slice for height and width
                vds_hyperslab = Hyperslab(
                    slice(block_start, block_start + block_size),
                    self.FULL_SLICE, self.FULL_SLICE)
                v_target = vds[vds_hyperslab.tuple]

                v_map = VirtualMap(v_source, v_target,
                                   dtype=source_meta.dtype)

                self.logger.debug("Mapping %s %s to %s of %s.",
                                  dataset.split("/")[-1], source_hyperslab,
                                  vds_hyperslab, self.name)
                map_list.append(v_map)

        return map_list
