import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
import multiprocessing
import time
import ctypes
import numpy as np
import torch

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
lib.prepare_threads.argtypes = [
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
lib.attn_output_threaded.restype = None
lib.set_ready_flag.argtypes = []
lib.set_ready_flag.restype = None

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
    def run_start_threads():
        lib.prepare_threads(
            ctypes.cast(values.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(logits.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(result.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(head_num),
            ctypes.c_int(batch_size),
            ctypes.c_int(K),
            ctypes.c_int(Dh),
            ctypes.c_int(kv_head_offset),
            ctypes.c_int(kv_batch_offset),
            ctypes.c_int(logits_score_head_offset),
            ctypes.c_int(logits_score_batch_offset),
            ctypes.c_int(q_out_head_offset),
            ctypes.c_int(q_out_batch_offset),
            ctypes.c_int(thread_num)
        )

    thread = threading.Thread(target=run_start_threads)
    thread.start()

    time.sleep(1)

    start_t = time.perf_counter_ns()
    lib.set_ready_flag()

    while not lib.is_finished():
        pass

    end_t = time.perf_counter_ns()
    duration = end_t - start_t
    print(f"Took {duration*1e-3} microseconds")

if __name__ == "__main__":
    batch_size = 16
    K = 81
    thread_num = 64
    Dh = 128
    head_num = 32

    # values = aligned_tensor(size=head_num * batch_size * K * Dh)
    # logits = aligned_tensor(size=head_num * batch_size * K)
    # result = aligned_tensor(size=head_num * batch_size * Dh)

    # values[:] = torch.rand(head_num * batch_size * K * Dh)
    # logits[:] = torch.rand(head_num * batch_size * K)
    # result[:] = torch.zeros(head_num * batch_size * Dh)

    values = torch.rand(head_num * batch_size * K * Dh)
    logits = torch.rand(head_num * batch_size * K)
    result = torch.zeros(head_num * batch_size * Dh)

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
