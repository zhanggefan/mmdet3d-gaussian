import pickle
import torch
from torch import distributed as dist
from mmcv.utils.progressbar import ProgressBar
import os
import sys
import itertools
from mmdet3d.utils import get_root_logger
import numpy as np
import mmap


class SharedList(list):
    store_count = itertools.count(0)

    def __init__(self, obj):
        assert isinstance(obj, list) or obj is None, "given object is not list"
        self.store = None
        if not dist.is_initialized():
            super(SharedList, self).__init__(obj)
            return

        super(SharedList, self).__init__()
        is_master = torch.cuda.current_device() == 0

        self.id = next(self.store_count)
        filename = f'/dev/shm/mmdet3d_tmplist_{self.id}'
        self.iter = 0

        if is_master:
            logger = get_root_logger()
            logger.info(f'serialize data to Shared MMAP of {filename}')
            with open(filename, 'wb') as f:
                idx = []
                prog_bar = ProgressBar(len(obj), 50, file=sys.stdout)
                for i, ele in enumerate(obj):
                    idx.append(f.tell())
                    f.write(pickle.dumps(ele, pickle.HIGHEST_PROTOCOL))
                    if (i + 1) % 100 == 0:
                        prog_bar.update(100)
                if (i + 1) % 100 != 0:
                    prog_bar.update((i + 1) % 100)
                idx.append(f.tell())
                prog_bar.file.write('\n')
            np.array(idx).astype(dtype=np.int64).tofile(filename + '.index')

        dist.barrier()
        fd = os.open(filename, os.O_RDONLY)
        self.store = mmap.mmap(fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
        self.index = np.fromfile(filename + '.index', dtype=np.int64)
        self.len = len(self.index) - 1

    def __getitem__(self, item):
        if self.store is None:
            return super(SharedList, self).__getitem__(item)
        if isinstance(item, (int, np.integer)):
            item = int(item)
            if not -self.len <= item < self.len:
                raise IndexError('list index out of range')
            item = item % self.len
            self.store.seek(self.index[item])
            read_len = self.index[item + 1] - self.index[item]
            return pickle.loads(self.store.read(read_len))
        elif isinstance(item, slice):
            start, stop, stride = item.indices(self.len)
            return [self.__getitem__(i) for i in range(start, stop, stride)]
        else:
            raise TypeError(
                f"list indices must be integers or slices, not {type(item)}")

    def __len__(self):
        if self.store is None:
            return super(SharedList, self).__len__()
        return self.len

    def __iter__(self):
        if self.store is None:
            return super(SharedList, self).__iter__()
        self.iter = 0
        return self

    def __next__(self):
        if self.store is None:
            return super(SharedList, self).__next__()
        if self.iter < self.len:
            ret = self.__getitem__(self.iter)
            self.iter += 1
            return ret
        else:
            raise StopIteration

    def __repr__(self):
        if self.__len__() == 0:
            return "SharedList([])"
        elif self.__len__() == 1:
            return f"SharedList([{str(self[0])}])"
        elif self.__len__() == 2:
            return f"SharedList([{str(self[0])}, {str(self[-1])}])"
        return f"SharedList([{str(self[0])}, ... {str(self[-1])}])"


class SharedDictOfList(dict):
    def __init__(self, obj):
        assert isinstance(obj, dict) or obj is None, "given object is not dict"
        self.store = None
        if not dist.is_initialized():
            super(SharedDictOfList, self).__init__(obj)
            return

        self.index = 0
        keys = None if obj is None else [*obj.keys()]
        keys = SharedList(keys)
        for k in keys:
            val_list = None if obj is None else obj[k]
            self[k] = SharedList(val_list)

    def __repr__(self):
        if self.__len__() == 0:
            return "SharedDictOfList({})"
        for key1 in self.keys():
            break
        if self.__len__() == 1:
            return "SharedDictOfList({" + f"{key1} : {str(self[key1])}" + "})"
        return "SharedDictOfList({" + f"{key1} : {str(self[key1])} ... " + "})"
