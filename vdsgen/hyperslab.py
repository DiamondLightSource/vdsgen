

class Hyperslab(object):

    """A 3D+ sub-section of an HDF5 file."""

    def __init__(self, frames, height, width):
        if isinstance(frames, list):
            self._frames = frames
        else:
            self._frames = [frames]
        self._height = height
        self._width = width

        self.tuple = tuple(self._frames + [height, width])

    def _format_slice(self, _slice):
        s = str(_slice)
        s = s.replace("slice", "")  # Don't print `slice`
        s = s.replace("(None, None, None)", "*")  # Entire extent of axis = `*`
        s = s.replace(", None)", ")")  # Don't print step of None
        return s

    def __str__(self):
        output_str = "{"
        for axis in self._frames:
            output_str += self._format_slice(axis) + ", "
        output_str += "{height}, {width}}}".format(
            height=self._format_slice(self._height),
            width=self._format_slice(self._width))

        return output_str
