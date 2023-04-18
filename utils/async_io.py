import collections
import queue
import sys
import typing

import torch
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import _wrap  # type: ignore[attr-defined]
from tqdm import tqdm


def batches_from_list(items: list[typing.Any], batch_size=512) -> list[list[typing.Any]]:
    return [items[i:(i + batch_size)] for i in range(0, len(items), batch_size)]


class Batch:

    @staticmethod
    def empty():
        return Batch(order=sys.maxsize, data=None, is_end=True)

    @staticmethod
    def from_batch(batch, data: typing.Any):
        return Batch(order=batch.order, data=data, is_end=False)

    def __init__(self, order: int, data: typing.Any = None, is_end: bool = False):
        self._order = order
        self._data = data
        self._is_end = is_end

    @property
    def order(self):
        return self._order

    @property
    def data(self):
        return self._data

    def _check_state(self):
        if self._is_end:
            raise ValueError(f"{self._is_end=}")

    def __bool__(self):
        return not self._is_end

    def __getitem__(self, item):
        self._check_state()
        return self._data.__getitem__(item)

    def __iter__(self):
        self._check_state()
        return self._data.__iter__()

    def __len__(self):
        self._check_state()
        return self._data.__len__()

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__str__()


_queue_t = queue.Queue
# signature of function should be (rank, queue_in, queue_out) -> bool
_async_fn_t = typing.Callable[[int, _queue_t, _queue_t], bool]


def start_processes(fn, args=(), nprocs=1, join=True, daemon=False, ctx=None):
    _mp = ctx or torch.multiprocessing.get_context('fork')
    error_queues = []
    processes = []
    for i in range(nprocs):
        error_queue = _mp.SimpleQueue()
        process = _mp.Process(
            target=_wrap,
            args=(fn, i, args, error_queue),
            daemon=daemon,
        )
        process.start()
        error_queues.append(error_queue)
        processes.append(process)

    context = torch.multiprocessing.ProcessContext(processes, error_queues)
    if not join:
        return context

    # Loop on join until it returns True or raises an exception.
    while not context.join():
        pass


def run_async_in_batches(items: list[typing.Any], batch_size, async_fn: _async_fn_t,
                         async_fn_kwargs: collections.OrderedDict = None, n_proc=torch.multiprocessing.cpu_count()):
    if n_proc == 0:
        raise ValueError(f"{n_proc=} is not supported!")

    batches = batches_from_list(items, batch_size)
    num_batches = len(batches)

    ctx = mp.get_context('spawn')  # 'fork'
    queue_in: queue.Queue = ctx.Queue(maxsize=128)
    queue_out = ctx.Queue(maxsize=128)  # mp.SimpleQueue()
    async_fn_kwargs = async_fn_kwargs or collections.OrderedDict()

    batch_counter = 0
    batch_queue = list(batches)
    batch_queue.extend([Batch.empty() for _ in range(n_proc)])
    batch_queue_size = num_batches + n_proc
    for batch_idx in tqdm(range(len(batches))):
        queue_in.put(Batch(batch_counter, batches[batch_idx]))
        batch_counter += 1
        if queue_in.full():
            break

    context = start_processes(
        async_fn,
        args=(queue_in, queue_out,) + tuple(async_fn_kwargs.values()),
        nprocs=n_proc,
        join=False,
        daemon=False
    )

    output = []
    sort = False

    with tqdm(
            total=num_batches, desc="queue_out") as tq:
        for _ in range(num_batches):
            infos = collections.OrderedDict()
            while batch_counter < (batch_queue_size) and not queue_in.full():
                queue_obj = batch_queue[batch_counter]
                if not isinstance(queue_obj, Batch):
                    queue_obj = Batch(batch_counter, queue_obj)
                queue_in.put(queue_obj)
                batch_counter += 1
            infos.update(batch_size=batch_size)
            infos.update(bc=batch_counter)

            batch_out = queue_out.get()
            if isinstance(batch_out, Batch):
                sort = True
                output.append((batch_out.order, batch_out.data))
            elif isinstance(batch_out, list) and isinstance(batch_out[0], Batch):
                sort = True
                for batch in batch_out:
                    output.append((batch.order, batch.data))
            else:
                if sort:
                    raise ValueError(f"{batch_out} is not a Batch but the previous ones were!")
                output.extend(batch_out)
            tq.update(1)
            tq.set_postfix(infos)

    if sort:
        output.sort(key=lambda x: x[0])
        output = [x for (_, batch) in output for x in batch]

    if context is not None:
        while context is not None and not context.join():
            pass
    return output
