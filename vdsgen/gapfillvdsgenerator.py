"""A class for generating virtual dataset frames from sub-frames."""

from vds import VirtualSource, VirtualLayout, h5slice

from .vdsgenerator import VDSGenerator, SourceMeta


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

        if len(self.files) > 1:
            raise ValueError("Can only insert gaps with a single dataset")

        self.source_file = self.files[0]  # We only have one dataset

    def process_source_datasets(self):
        """Grab data from the given HDF5 files and check for consistency.

        Returns:
            Source: Number of datasets and the attributes of them (frames,
                height width and data type)

        """
        data = self.grab_metadata(self.files[0])

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

    def create_virtual_layout(self, source_meta):
        """Create a VirtualLayout mapping raw data to the VDS.

        Args:
            source_meta(SourceMeta): Source attributes

        Returns:
            VirtualLayout: Object describing links between raw data and VDS

        """
        x_spacing, y_spacing = self.construct_vds_spacing()

        target_shape = source_meta.frames + \
            (source_meta.height + sum(y_spacing),
             source_meta.width + sum(x_spacing))
        self.logger.debug("VDS metadata:\n"
                          "  Shape: %s\n", target_shape)

        v_layout = VirtualLayout(target_shape, source_meta.dtype)

        source_shape = source_meta.frames + \
            (source_meta.height, source_meta.width)
        v_source = VirtualSource(
            self.source_file, name=self.source_node,
            shape=source_shape, dtype=source_meta.dtype
        )

        map_frames = max(source_meta.frames[0] // 100, 1)
        for map_idx, _ in enumerate(range(0, source_meta.frames[0], map_frames)):
            frame_start = map_idx * map_frames
            frame_end = min(frame_start + map_frames, source_meta.frames[0])
            y_current = 0
            for module_idx in range(self.grid_y / 2):
                y_start = y_current
                y_stop = y_start + self.sub_height * 2 + y_spacing[0]
                y_current = y_stop + y_spacing[1]

                source_y_start = self.sub_height * module_idx * 2
                source_y_end = source_y_start + self.sub_height * 2

                # Hyperslab: All frames,
                #            Height bounds of module,
                #            Full width
                source_hyperslab = v_source[frame_start:frame_end, source_y_start:source_y_end, :]

                # Hyperslab: All frames,
                #            Selection of chips with vertical gaps,
                #            Selection of chips with horizontal gaps
                v_layout[
                    frame_start:frame_end,
                    h5slice(y_start, 2, self.sub_height + y_spacing[0], self.sub_height),
                    h5slice(0, 8, self.sub_width + x_spacing[0], self.sub_width)
                ] = source_hyperslab

                self.logger.debug(
                    "Mapping %s[%s:%s, %s:%s, :] to %s[%s:%s, %s:%s, :].",
                    self.name, frame_start, frame_end, source_y_start, source_y_end,
                    self.source_file.split("/")[-1], frame_start, frame_end, y_start, y_stop)

        return v_layout
