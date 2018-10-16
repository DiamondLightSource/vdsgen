from h5py import h5s
from h5py import VirtualLayout
from h5py._hl.vds import VDSmap
from h5py._hl.selections import SimpleSelection, \
    _expand_ellipsis, _translate_int, _translate_slice


class AdvancedVirtualLayout(VirtualLayout):

    def __setitem__(self, key, source):
        if any([isinstance(element, tuple) for element in key]):
            start, count, stride, block = _handle_advanced(self.shape, key)
            src_dspace = h5s.create_simple(source.shape, source.maxshape)
            src_dspace.select_hyperslab(
                start=start, count=count, stride=stride, block=block
            )
        else:
            src_dspace = SimpleSelection(self.shape)[key].id

        self.sources.append(
            VDSmap(src_dspace, source.path, source.name, source.sel.id)
        )


def _handle_advanced(shape, args):
    start = []
    count = []
    stride = []
    block = []

    args = _expand_ellipsis(args, len(shape))

    for arg, length in zip(args, shape):
        if isinstance(arg, int):
            _start, _count, _stride = _translate_int(arg, length)
            _block = 1
        elif isinstance(arg, slice):
            _start, _count, _stride = _translate_slice(arg, length)
            _block = 1
        elif isinstance(arg, tuple):
            if len(arg) != 4:
                raise KeyError(
                    "Advanced slicing requires a 4-tuple: "
                    "(start, count, stride, block)"
                )
            else:
                _start, _count, _stride, _block = arg
        else:
            raise KeyError("Selection element must be int, slice or tuple")

        start.append(_start)
        count.append(_count)
        stride.append(_stride)
        block.append(_block)

    return tuple(start), tuple(count), tuple(stride), tuple(block)
