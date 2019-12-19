import itertools
from multiprocessing import Process, Queue

import math
from tqdm import tqdm

from dhfcorr.utils import batch


def process_multiple_data_in_one_run(results, data, function, **kwargs):
    process_results = [function(d, **kwargs) for d in data]
    results.put(process_results)


def process_multicore(worker, list_of_data, n_treads, message=None, **kwargs):
    data_blocks = list(batch(list_of_data, math.ceil(len(list_of_data) / n_treads)))
    queue = Queue()
    processes = list()

    for data_worker in data_blocks:
        p = Process(target=lambda x: process_multiple_data_in_one_run(queue, x, worker, **kwargs),
                    args=(data_worker,))
        processes.append(p)
        p.start()

    combined_results = list(itertools.chain.from_iterable([queue.get() for _ in tqdm(range(len(data_blocks)),
                                                                                     total=len(data_blocks),
                                                                                     desc=message)]))
    for p in processes:
        p.join()
    return combined_results
