import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
import multiprocessing
import time
import ctypes
import numpy as np
import torch
import argparse
import os

ITER = 10


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


def aligned_tensor(size, dtype=torch.float32, alignment=32):
    aligned_array_np = aligned_array(size, dtype=np.float32, alignment=alignment)
    return torch.from_numpy(aligned_array_np).float()


# Load the shared library
lib = ctypes.CDLL("./libattn.so")

# Define the function prototypes
lib.prepare_value_gemv.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # values
    ctypes.POINTER(ctypes.c_float),  # logits
    ctypes.POINTER(ctypes.c_float),  # result
    ctypes.c_int,  # head_num
    ctypes.c_int,  # batch_size
    ctypes.c_int,  # K
    ctypes.c_int,  # Dh
    ctypes.c_int,  # values_head_offset
    ctypes.c_int,  # values_batch_offset
    ctypes.c_int,  # logits_head_offset
    ctypes.c_int,  # logits_batch_offset
    ctypes.c_int,  # result_head_offset
    ctypes.c_int,  # result_batch_offset
    ctypes.c_int,  # num_threads
    # ctypes.POINTER(ctypes.c_float) # done
]
lib.prepare_value_gemv.restype = None
lib.prepare_key_gemv.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # keys
    ctypes.POINTER(ctypes.c_float),  # queries
    ctypes.POINTER(ctypes.c_float),  # logits
    ctypes.c_int,  # head_num
    ctypes.c_int,  # batch_size
    ctypes.c_int,  # K
    ctypes.c_int,  # Dh
    ctypes.c_int,  # values_head_offset
    ctypes.c_int,  # values_batch_offset
    ctypes.c_int,  # logits_head_offset
    ctypes.c_int,  # logits_batch_offset
    ctypes.c_int,  # result_head_offset
    ctypes.c_int,  # result_batch_offset
    ctypes.c_int,  # num_threads
    # ctypes.POINTER(ctypes.c_float) # done
]
lib.prepare_key_gemv.restype = None
lib.set_ready_flag.argtypes = []
lib.set_ready_flag.restype = None
lib.is_finished.argtypes = []
# lib.is_finished.restype = ctypes.c_double
lib.is_finished.restype = ctypes.c_long
lib.clear_flags.argtypes = []
lib.clear_flags.restype = None
lib.get_duration.argtypes = []
lib.get_duration.restype = ctypes.c_double
lib.wait_finished.argtypes = []
lib.wait_finished.restype = None


def test_with_threading(
    batch_size,
    K,
    thread_num,
    Dh,
    head_num,
    values_keys,
    logits_queries,
    result_logits,
    kv_head_offset,
    kv_batch_offset,
    logits_queries_head_offset,
    logits_queries_batch_offset,
    out_logits_head_offset,
    out_logits_batch_offset,
    is_key_gemv,
):
    def task_value_gemv():
        # Elevate the process to the highest priority(real-time class)
        sched_param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, sched_param)
        except PermissionError:
            print("Permission denied. Try running as root.")
        
        os.sched_setaffinity(0, {5})
        lib.prepare_value_gemv(
            # ctypes.cast(values_keys.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            # ctypes.cast(logits_queries.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            # ctypes.cast(result_logits.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            values_keys.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            logits_queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            result_logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(head_num),
            ctypes.c_int(batch_size),
            ctypes.c_int(K),
            ctypes.c_int(Dh),
            ctypes.c_int(kv_head_offset),
            ctypes.c_int(kv_batch_offset),
            ctypes.c_int(logits_queries_head_offset),
            ctypes.c_int(logits_queries_batch_offset),
            ctypes.c_int(out_logits_head_offset),
            ctypes.c_int(out_logits_batch_offset),
            ctypes.c_int(thread_num),
        )

    def task_key_gemv():
        # Elevate the process to the highest priority(real-time class)
        sched_param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, sched_param)
        except PermissionError:
            print("Permission denied. Try running as root.")

        os.sched_setaffinity(0, {5})
        lib.prepare_key_gemv(
            ctypes.cast(values_keys.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(logits_queries.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(result_logits.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(head_num),
            ctypes.c_int(batch_size),
            ctypes.c_int(K),
            ctypes.c_int(Dh),
            ctypes.c_int(kv_head_offset),
            ctypes.c_int(kv_batch_offset),
            ctypes.c_int(logits_queries_head_offset),
            ctypes.c_int(logits_queries_batch_offset),
            ctypes.c_int(out_logits_head_offset),
            ctypes.c_int(out_logits_batch_offset),
            ctypes.c_int(thread_num),
        )

    if is_key_gemv:
        thread = threading.Thread(target=task_key_gemv)
    else:
        thread = threading.Thread(target=task_value_gemv)
    thread.start()

    time.sleep(1)
    # dummy = 0
    # while dummy < 100000:
    #     dummy += 1

    print("=====================================================")
    start_t = time.perf_counter_ns()
    lib.set_ready_flag()

    # while not lib.is_finished():
    #     pass
    # fin = lib.is_finished()

    lib.wait_finished()
    end_t = time.perf_counter_ns()
    duration = (end_t - start_t) / 1e3
    # duration = lib.get_duration()
    # duration = lib.is_finished() / 1e3
    print(f"Took {duration} microseconds")
    lib.clear_flags()
    thread.join()

def aligned_array(shape, dtype, alignment=64):
    """Create a memory-aligned numpy array."""
    nbytes = np.prod(shape) * np.dtype(dtype).itemsize
    buffer = np.empty(nbytes + alignment, dtype=np.uint8)
    start_index = -buffer.ctypes.data % alignment
    aligned_buffer = buffer[start_index:start_index + nbytes]
    return np.ndarray(shape, dtype=dtype, buffer=aligned_buffer)

def test_value_gemv(batch_size, K, thread_num):
    # Elevate the process to the highest priority(real-time class)
    sched_param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, sched_param)
    except PermissionError:
        print("Permission denied. Try running as root.")

    # pid = os.getpid()
    # try:
    #     os.sched_setaffinity(pid, [49])
    # except AttributeError:
    #     print("os.sched_setaffinity is not available on this platform.")

    # try:
    #     os.nice(-20)
    # except PermissionError:
    #     print("Permission denied. Try running as root.")

    head_num = 32
    Dh = 128

    # Memory-aligned allocation using Torch
    # values = torch.rand(head_num * batch_size * K * Dh, dtype=torch.float32)
    # logits = torch.rand(head_num * batch_size * K, dtype=torch.float32)
    # result = torch.zeros(head_num * batch_size * Dh, dtype=torch.float32)

    # Check memory alignment (optional)
    # assert values.data_ptr() % 64 == 0, "values is not 64-byte aligned!"
    # assert logits.data_ptr() % 64 == 0, "logits is not 64-byte aligned!"
    # assert result.data_ptr() % 64 == 0, "result is not 64-byte aligned!"

    # Memory-aligned allocation using NumPy (default alignment is sufficient for most cases)
    values = aligned_array((head_num, batch_size, K, Dh), dtype=np.float32, alignment=64)
    logits = aligned_array((head_num, batch_size, K, Dh), dtype=np.float32, alignment=64)
    result = aligned_array((head_num, batch_size, K, Dh), dtype=np.float32, alignment=64)

    # # Fill values and logits with random values
    values[:] = np.random.rand(*values.shape).astype(np.float32)
    logits[:] = np.random.rand(*logits.shape).astype(np.float32)
    
    # Ensure alignment (numpy arrays are typically well-aligned for SIMD operations)
    assert values.ctypes.data % 64 == 0, "values is not 64-byte aligned!"
    assert logits.ctypes.data % 64 == 0, "logits is not 64-byte aligned!"
    assert result.ctypes.data % 64 == 0, "result is not 64-byte aligned!"

    kv_head_offset = batch_size * K * Dh
    kv_batch_offset = K * Dh
    logits_queries_head_offset = batch_size * K
    logits_queries_batch_offset = K
    out_logits_head_offset = batch_size * Dh
    out_logits_batch_offset = Dh

    for _ in range(ITER):
        test_with_threading(
            batch_size,
            K,
            thread_num,
            Dh,
            head_num,
            values,
            logits,
            result,
            kv_head_offset,
            kv_batch_offset,
            logits_queries_head_offset,
            logits_queries_batch_offset,
            out_logits_head_offset,
            out_logits_batch_offset,
            False,
        )
        # values = torch.rand(head_num * batch_size * K * Dh, dtype=torch.float32)
        # logits = torch.rand(head_num * batch_size * K, dtype=torch.float32)
        # result = torch.zeros(head_num * batch_size * Dh, dtype=torch.float32)
        # assert values.data_ptr() % 64 == 0, "values is not 64-byte aligned!"
        # assert logits.data_ptr() % 64 == 0, "logits is not 64-byte aligned!"
        # assert result.data_ptr() % 64 == 0, "result is not 64-byte aligned!"
        # Memory-aligned allocation using NumPy (default alignment is sufficient for most cases)
        values = aligned_array((head_num, batch_size, K, Dh), dtype=np.float32, alignment=64)
        logits = aligned_array((head_num, batch_size, K, Dh), dtype=np.float32, alignment=64)
        result = aligned_array((head_num, batch_size, K, Dh), dtype=np.float32, alignment=64)

        # # Fill values and logits with random values
        values[:] = np.random.rand(*values.shape).astype(np.float32)
        logits[:] = np.random.rand(*logits.shape).astype(np.float32)
        
        # Ensure alignment (numpy arrays are typically well-aligned for SIMD operations)
        assert values.ctypes.data % 64 == 0, "values is not 64-byte aligned!"
        assert logits.ctypes.data % 64 == 0, "logits is not 64-byte aligned!"
        assert result.ctypes.data % 64 == 0, "result is not 64-byte aligned!"


def test_key_gemv(batch_size, K, thread_num):
    head_num = 4
    Dh = 128

    kv_head_offset = batch_size * K * Dh
    kv_batch_offset = K * Dh
    queries_head_offset = batch_size * Dh
    queries_batch_offset = Dh
    logits_head_offset = batch_size * K
    logits_batch_offset = K

    keys = torch.rand(head_num * batch_size * K * Dh)
    queries = torch.rand(head_num * batch_size * Dh)
    logits = torch.zeros(head_num * batch_size * K)

    for _ in range(ITER):
        test_with_threading(
            batch_size,
            K,
            thread_num,
            Dh,
            head_num,
            keys,
            queries,
            logits,
            kv_head_offset,
            kv_batch_offset,
            queries_head_offset,
            queries_batch_offset,
            logits_head_offset,
            logits_batch_offset,
            True,
        )
        keys = torch.rand(head_num * batch_size * K * Dh)
        queries = torch.rand(head_num * batch_size * Dh)
        logits = torch.zeros(head_num * batch_size * K)


def check_power_of_2(value):
    if value <= 0 or (value & (value - 1)) != 0:
        raise argparse.ArgumentTypeError(
            f"In current implementation, K in Key GEMV should be the power of 2, but {value} is not a power of 2"
        )
    return value


def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="A program to test running C++ GEMV kernels in a Python code."
    )

    parser.add_argument("--batch_size", type=int, default=32, help="Number of batches")
    parser.add_argument("--K", type=int, default=81, help="Number of topk indices")
    parser.add_argument("--thread_num", type=int, default=48, help="Number of threads")
    parser.add_argument("--key_gemv", action="store_true", help="Test Key GEMV")

    args = parser.parse_args()

    if args.key_gemv:
        print("here")
        check_power_of_2(args.K)
        test_key_gemv(args.batch_size, args.K, args.thread_num)
    else:
        test_value_gemv(args.batch_size, args.K, args.thread_num)


if __name__ == "__main__":
    os.sched_setaffinity(0, {3})
    main()
