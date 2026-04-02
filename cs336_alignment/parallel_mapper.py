from typing import Callable, Optional

from multiprocess import cpu_count, get_context


class ParallelMapper:
    def __init__(
        self,
        func: Callable,
        processes: Optional[int] = None,
        chunksize: Optional[int] = None,
    ):
        self.func = func
        self.processes = processes or cpu_count()
        self.chunksize = chunksize
        ctx = get_context("spawn")
        self.pool = ctx.Pool(processes=self.processes)

    def map(self, *iterable):
        data = list(zip(*iterable))
        if self.chunksize is None:
            chunksize = max(1, len(data) // (self.processes * 4))
        else:
            chunksize = self.chunksize
        return self.pool.starmap(self.func, data, chunksize)

    def close(self):
        self.pool.close()
        self.pool.join()
