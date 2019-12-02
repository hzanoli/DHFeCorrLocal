from multiprocessing import Process, Queue
from dhfcorr.io.utils import batch
import itertools
import math


def process_multiple_data_in_one_run(results, data, function, **kwargs):
    process_results = [function(d, **kwargs) for d in data]
    results.put(process_results)


def process(worker, list_of_data, n_treads, **kwargs):
    data_blocks = list(batch(list_of_data, math.ceil(len(list_of_data) / n_treads)))
    queue = Queue()
    processes = list()

    for data_worker in data_blocks:
        p = Process(target=lambda x: process_multiple_data_in_one_run(queue, x, worker, **kwargs),
                    args=(data_worker,))
        processes.append(p)
        p.start()

    combined_results = list(itertools.chain.from_iterable([queue.get() for x in range(len(data_blocks))]))

    for p in processes:
        p.join()

    return combined_results
