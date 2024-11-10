import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
import multiprocessing
import time
import numpy as np

import build.attn_module as attn_module

# import attn_module

import ctypes
from utils import set_process_priority, set_cpu_affinity
import os


def aligned_array(size, dtype=np.float32, alignment=32):
    # Determine the number of bytes per element
    itemsize = np.dtype(dtype).itemsize
    # Allocate raw memory with alignment
    buf = ctypes.create_string_buffer(size * itemsize + alignment)
    start = ctypes.addressof(buf)
    offset = (alignment - start % alignment) % alignment
    # Create a numpy array that views this aligned memory
    aligned_array = np.frombuffer(buf, dtype=dtype, count=size, offset=offset)

    return aligned_array


# Sample task_thread_pool for each thread
def task_thread_pool(
    values,
    logits,
    result,
    head_num,
    batch_size,
    K,
    Dh,
    kv_head_offset,
    kv_batch_offset,
    logits_score_head_offset,
    logits_score_batch_offset,
    q_out_head_offset,
    q_out_batch_offset,
    t_idx,
    thread_num,
    start_idx,
    end_idx,
    ready_flag,
    finished_flag,
):
    # Busy-wait until the ready_flag is set
    while not ready_flag.is_set():
        pass  # Keep checking the flag status without releasing CPU
    attn_module.attn_output_threaded(
        values,
        logits,
        result,
        head_num=head_num,
        batch_size=batch_size,
        K=K,
        Dh=Dh,
        values_head_offset=kv_head_offset,
        values_batch_offset=kv_batch_offset,
        logits_head_offset=logits_score_head_offset,
        logits_batch_offset=logits_score_batch_offset,
        result_head_offset=q_out_head_offset,
        result_batch_offset=q_out_batch_offset,
        thread_id=t_idx,
        thread_num=thread_num,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    # finished_flag.set()
    return time.perf_counter_ns()


def test_thread_pool(
    batch_size,
    K,
    thread_num,
    Dh,
    head_num,
    values,
    logits,
    result,
    total_work,
    work_per_thread,
    kv_head_offset,
    kv_batch_offset,
    logits_score_head_offset,
    logits_score_batch_offset,
    q_out_head_offset,
    q_out_batch_offset,
):
    # Initialize the ready_flag as a threading.Event
    ready_flag = threading.Event()
    finish_flags = [threading.Event() for _ in range(thread_num)]
    # Number of threads
    data_list = [
        (
            values,
            logits,
            result,
            head_num,
            batch_size,
            K,
            Dh,
            kv_head_offset,
            kv_batch_offset,
            logits_score_head_offset,
            logits_score_batch_offset,
            q_out_head_offset,
            q_out_batch_offset,
            t_idx,
            thread_num,
            t_idx * work_per_thread,
            min(t_idx * work_per_thread + work_per_thread, total_work),
            ready_flag,
            finish_flags[t_idx],
        )
        for t_idx in range(thread_num)
    ]

    # Measure the latency
    # Initialize ThreadPool
    with ThreadPool(thread_num) as pool:

        # Submit tasks to ThreadPool
        start_t = time.perf_counter_ns()
        result = pool.starmap_async(task_thread_pool, data_list)
        # time.sleep(1)
        # ready_flag.set()

        # all_threads_finished = False
        # while not all_threads_finished:
        #     all_threads_finished = True
        #     for finished_flag in finish_flags:
        #         if not finished_flag.is_set():
        #             all_threads_finished = False
        #             break
        # pool.close()
        # end_t = time.perf_counter_ns()
        # pool.join()
        for end_t in result.get():
            duration = end_t - start_t
            print(f"Took {duration*1e-3} microseconds")
    print("========================================================")

    # duration = end_t - start_t
    # print(f'Took {duration*1e-3} microseconds')


# Sample multiprocessing function for each thread
def task_multiprocessing(
    values,
    logits,
    result,
    head_num,
    batch_size,
    K,
    Dh,
    kv_head_offset,
    kv_batch_offset,
    logits_score_head_offset,
    logits_score_batch_offset,
    q_out_head_offset,
    q_out_batch_offset,
    t_idx,
    thread_num,
    start_idx,
    end_idx,
    ready_flag,
    finished_flag,
):
    pid = os.getpid()
    set_process_priority(pid, -20)
    cpu = 48 + (t_idx - 23) if t_idx > 23 else t_idx

    set_cpu_affinity(pid, [cpu])

    # Busy-wait until the ready_flag is set
    # while not ready_flag.value:
    #     pass  # Keep checking the flag status without releasing CPU
    # while True:
    #     if ready_flag.value:
    #         break

    start_t = time.perf_counter_ns()

    # ready_flag_obj = ready_flag.get_obj()
    attn_module.attn_output_threaded(
        values,
        logits,
        result,
        head_num=head_num,
        batch_size=batch_size,
        K=K,
        Dh=Dh,
        values_head_offset=kv_head_offset,
        values_batch_offset=kv_batch_offset,
        logits_head_offset=logits_score_head_offset,
        logits_batch_offset=logits_score_batch_offset,
        result_head_offset=q_out_head_offset,
        result_batch_offset=q_out_batch_offset,
        thread_id=t_idx,
        thread_num=thread_num,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    end_t = time.perf_counter_ns()
    print(f"Duration: {(end_t - start_t)*1e-3}")
    finished_flag.value = True
    # return time.perf_counter_ns()


def test_multiprocessing(
    batch_size,
    K,
    thread_num,
    Dh,
    head_num,
    values,
    logits,
    result,
    total_work,
    work_per_thread,
    kv_head_offset,
    kv_batch_offset,
    logits_score_head_offset,
    logits_score_batch_offset,
    q_out_head_offset,
    q_out_batch_offset,
):
    # ready_flag = multiprocessing.Value(ctypes.c_bool, False)
    ready_flag = ctypes.c_bool(False)
    # ready_flag = False
    # finished_flags = [
    #     multiprocessing.Value(ctypes.c_bool, False) for _ in range(thread_num)
    # ]

    processes = []
    for t_idx in range(thread_num):
        start_idx = t_idx * work_per_thread
        end_idx = min(start_idx + work_per_thread, total_work)

        # Create a new process
        process = multiprocessing.Process(
            target=task_multiprocessing,
            args=(
                values,
                logits,
                result,
                head_num,
                batch_size,
                K,
                Dh,
                kv_head_offset,
                kv_batch_offset,
                logits_score_head_offset,
                logits_score_batch_offset,
                q_out_head_offset,
                q_out_batch_offset,
                t_idx,
                thread_num,
                start_idx,
                end_idx,
                ready_flag,
                # finished_flags[t_idx],
            ),
        )

        process.start()

        # Add process to the list for tracking
        processes.append(process)
    start_t = time.perf_counter_ns()
    ready_flag = True
    # all_threads_finished = False
    # while not all_threads_finished:
    #     all_threads_finished = True
    #     for finished_flag in finished_flags:
    #         if not finished_flag.value:
    #             all_threads_finished = False
    #             break
    end_t = time.perf_counter_ns()
    duration = end_t - start_t
    print(f"Took {duration*1e-3} microseconds")

    for process in processes:
        process.join()


if __name__ == "__main__":
    # Initialize example data
    batch_size = 128
    K = 128
    thread_num = 48
    Dh = 128
    head_num = 32

    values = aligned_array(size=head_num * batch_size * K * Dh)
    logits = aligned_array(size=head_num * batch_size * K)
    result = aligned_array(size=head_num * batch_size * Dh)
    values[:] = np.random.rand(head_num * batch_size * K * Dh)
    logits[:] = np.random.rand(head_num * batch_size * K)
    result[:] = np.zeros(head_num * batch_size * Dh)

    total_work = head_num * batch_size
    work_per_thread = int(total_work / thread_num)

    kv_head_offset = batch_size * K * Dh
    kv_batch_offset = K * Dh
    logits_score_head_offset = batch_size * K
    logits_score_batch_offset = K
    q_out_head_offset = batch_size * Dh
    q_out_batch_offset = Dh

    test_multiprocessing(
        batch_size,
        K,
        thread_num,
        Dh,
        head_num,
        values,
        logits,
        result,
        total_work,
        work_per_thread,
        kv_head_offset,
        kv_batch_offset,
        logits_score_head_offset,
        logits_score_batch_offset,
        q_out_head_offset,
        q_out_batch_offset,
    )

    # test_thread_pool(
    #     batch_size,
    #     K,
    #     thread_num,
    #     Dh,
    #     head_num,
    #     values,
    #     logits,
    #     result,
    #     total_work,
    #     work_per_thread,
    #     kv_head_offset,
    #     kv_batch_offset,
    #     logits_score_head_offset,
    #     logits_score_batch_offset,
    #     q_out_head_offset,
    #     q_out_batch_offset,
    # )
