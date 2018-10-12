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

        # Create a mixed radix set mapping any 1D index to an ND index
        # The 1D index is a decimal number and the ND index is the equivalent
        # representation in the mixed radix numeral system derived from shape
        # e.g. for shape (5, 3, 10) radices = 30, 10 and 1
        #   132 -> 412 (30*4 + 10*1 + 1*2) so [132] in 1D -> [4, 1, 2] in 3D
        radices = [1]  # Smallest radix is always worth 1 in decimal
        for axis_length in reversed(shape[1:]):
            radices.insert(0, radices[0] * axis_length)
        self.radices = tuple(radices)

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
        """Create a list of VirtualMaps of raw data to the VDS.

        Args:
            source_meta(SourceMeta): Source attributes

        Returns:
            VirtualLayout: Object describing links between raw data and VDS

        """
        vds_shape = self.dimensions + (source_meta.height, source_meta.width)
        self.logger.debug("VDS metadata:\n"
                          "  Shape: %s\n", vds_shape)
        v_layout = h5.VirtualLayout(vds_shape, source_meta.dtype)

        with h5.File(self.source_file) as source_file:
            v_source = h5.VirtualSource(source_file[self.source_node])

        self.calculate_axis_periods()

        # Iterate over total number of row hyperslabs to map to VDS
        for row_idx in range(self.calculate_required_maps()):
            lower_idx = row_idx * self.dimensions[-1]
            upper_idx = lower_idx + self.dimensions[-1]
            if self.alternate is not None and \
                    self.alternate[-1] and row_idx % 2:
                start = upper_idx - 1
                end = lower_idx - 1
                source_hyperslab = v_source[start:end:-1]
            else:
                start = lower_idx
                end = upper_idx
                source_hyperslab = v_source[start:end]

            axis_indices = self.calculate_axis_indices(lower_idx)
            axis_indices[-1] = self.FULL_SLICE
            # Hyperslab: Single index for each inner axis,
            #            Full extent of outermost axis,
            #            Full slice for height and width
            vds_hyperslab = tuple(axis_indices +
                                  [self.FULL_SLICE, self.FULL_SLICE])
            v_layout[vds_hyperslab] = source_hyperslab

            self.logger.debug(
                "Mapping %s[%s, 0:%s, ...] to %s[%s:%s, ...].",
                self.name, ", ".join(str(idx) for idx in axis_indices[:-1]),
                self.dimensions[-1],
                self.source_file.split("/")[-1], start, end)

        return v_layout

    def calculate_axis_periods(self):
        """Calculate cumulative periods for set of nested axes.

        An axis period is the total number of underlying steps needed to
        increment that axis by one index.

        Returns:
            list: Cumulative periods

        """
        period = 1
        # Iterate dimensions backwards excluding outermost
        for axis_length in self.dimensions[:0:-1]:
            # Period of a given axis is cumulative period of inner dimensions
            period *= axis_length
            self.periods.insert(0, period)

    def calculate_required_maps(self):
        """Calculate total individual maps required to generate the VDS.

        Returns:
            int: Required number of maps

        """
        return self.product(self.dimensions[:-1])

    @staticmethod
    def product(iterable):
        """Calculate product of elements of an iterable.

        Args:
            iterable: An object capable of returning its members one at a time.
                Must have at least on element.

        Returns:
            int: Product

        """
        product = 1
        for value in iterable:
            product *= value
        return product

    def calculate_axis_indices(self, frame_index):
        """Calculate indices for each inner axis for this frame index.

        Args:
            frame_index: Frame index in overall dataset

        Returns:
            tuple: Indices for each individual axis

        """
        axis_indices = [0 for _ in self.dimensions]
        for idx, radix in enumerate(self.radices):
            while frame_index >= radix:
                frame_index -= radix
                axis_indices[idx] += 1

        return axis_indices
