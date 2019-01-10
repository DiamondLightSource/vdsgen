"""A class for generating virtual dataset frames from sub-frames."""

import h5py as h5

from .vdsgenerator import VDSGenerator, SourceMeta


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

        super(InterleaveVDSGenerator, self).__init__(
            path, prefix, files, output, source, source_node, target_node,
            fill_value, log_level)

    def process_source_datasets(self):
        """Grab data from the given HDF5 files and check for consistency.

        Returns:
            Source: Number of datasets and the attributes of them (frames,
                height width and data type)

        """
        data = self.grab_metadata(self.files[0])
        frames = [data["frames"][0]]
        for dataset in self.files[1:]:
            temp_data = self.grab_metadata(dataset)
            frames.append(temp_data["frames"][0])
            for attribute, value in data.items():
                if attribute != "frames" and temp_data[attribute] != value:
                    raise ValueError("Files have mismatched "
                                     "{}".format(attribute))

        source = SourceMeta(frames=tuple(frames),
                            height=data['height'], width=data['width'],
                            dtype=data['dtype'])

        self.logger.debug("Source metadata retrieved:\n"
                          "  Frames: %s\n"
                          "  Height: %s\n"
                          "  Width: %s\n"
                          "  Data Type: %s", frames, *source[1:])
        return source

    def process_source_metadata(self, source):
        frames, height, width = self.parse_shape(source["shape"])
        if not isinstance(frames[0], tuple):
            raise ValueError(
                "For InterleaveVDSGenerator, source.shape.frames must be a "
                "tuple of the number of frames in each raw file."
            )
        source_metadata = SourceMeta(
            frames=frames[0], height=height, width=width,
            dtype=source["dtype"])

        return source_metadata

    def create_virtual_layout(self, source_meta):
        """Create a VirtualLayout mapping raw data to the VDS.

        Args:
            source_meta(SourceMeta): Source attributes

        Returns:
            VirtualLayout: Object describing links between raw data and VDS

        """
        total_frames = sum(source_meta.frames)
        target_shape = (total_frames,) + \
                       (source_meta.height, source_meta.width)
        self.logger.debug("VDS metadata:\n"
                          "  Shape: %s\n", target_shape)

        v_layout = h5.VirtualLayout(target_shape, source_meta.dtype)

        total_files = len(self.files)
        for file_idx, file_path in enumerate(self.files):
            source_shape = (source_meta.frames[file_idx],) + \
                (source_meta.height, source_meta.width)
            v_source = h5.VirtualSource(
                file_path, name=self.source_node,
                shape=source_shape, dtype=source_meta.dtype
            )
            dataset_frames = v_source.shape[0]

            for frame_idx in range(0, dataset_frames, self.block_size):

                source_block_idx = frame_idx // self.block_size
                target_block_idx = \
                    file_idx + total_files * source_block_idx

                # Make sure to only take as many frames as raw dataset has
                # It is OK if this isn't a complete block if it is the last one
                block_size = min(self.block_size, dataset_frames - frame_idx)

                if block_size < self.block_size:
                    self.logger.warning(
                        "%s does not have integer number of blocks sized %s",
                        file_path.split("/")[-1], self.block_size)

                source_start = frame_idx
                source_end = frame_idx + block_size
                # Hyperslab: Frame[block_start_idx, block_end_idx + 1],
                #            Full slice for height and width
                source_hyperslab = v_source[source_start: source_end, :, :]

                target_start = self.block_size * target_block_idx
                if target_start + block_size > total_frames:
                    raise RuntimeError(
                        "Cannot map dataset %d of %d [%d frames] "
                        "to vds [%d frames] with blocks sized %d" %
                        (self.files.index(file_path) + 1, len(self.files),
                         dataset_frames, total_frames, self.block_size))
                target_end = target_start + block_size

                # Hyperslab: Frame[block_start_idx, block_end_idx + 1],
                #            Full slice for height and width
                v_layout[target_start:target_end, :, :] = source_hyperslab

                self.logger.debug(
                    "Mapping %s[%s:%s, :, :] to %s[%s:%s, :, :].",
                    self.name, target_start, target_end,
                    file_path.split("/")[-1], source_start, source_end)

        return v_layout
