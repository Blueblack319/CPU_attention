import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
import multiprocessing
import time
import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import os
import random
from utils import (
    layout_1_value,
    layout_2_value,
    layout_3_value,
    layout_1_key,
    layout_2_key,
)

ITER = 10


def aligned_array(size, dtype=np.float16, alignment=32):
    # Determine the number of bytes per element
    itemsize = np.dtype(dtype).itemsize
    # Allocate raw memory with alignment
    buf = ctypes.create_string_buffer(size * itemsize + alignment)
    start = ctypes.addressof(buf)
    offset = (alignment - start % alignment) % alignment
    # Create a numpy array that views this aligned memory
    aligned_array = np.frombuffer(buf, dtype=dtype, count=size, offset=offset)

    return aligned_array


def aligned_tensor(size, dtype=torch.float16, alignment=32):
    aligned_array_np = aligned_array(size, dtype=np.float16, alignment=alignment)
    return torch.from_numpy(aligned_array_np).float()


# Load the shared library
lib = ctypes.CDLL("/home2/jaehoon/CPU_attention/integrate_with_ctypes/libattn.so")

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
]
lib.prepare_value_gemv.restype = None
lib.prepare_value_gemv_half.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # values
    ctypes.POINTER(ctypes.c_uint16),  # logits
    ctypes.POINTER(ctypes.c_uint16),  # result
    ctypes.POINTER(ctypes.c_uint16),  # topk_indices
    ctypes.c_int,  # topk_num
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
    ctypes.c_int,  # total_work
]
lib.prepare_value_gemv_half.restype = None
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
]
lib.prepare_key_gemv.restype = None
lib.prepare_key_gemv_half.argtypes = [
    ctypes.POINTER(ctypes.c_uint16),  # keys
    ctypes.POINTER(ctypes.c_uint16),  # queries
    ctypes.POINTER(ctypes.c_uint16),  # logits
    ctypes.POINTER(ctypes.c_uint16),  # topk_indices
    ctypes.c_int,  # topk_num
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
    ctypes.c_int,  # total_work
]
lib.prepare_key_gemv_half.restype = None
lib.prepare_softmax.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # qk
    ctypes.POINTER(ctypes.c_float),  # max_values
    ctypes.POINTER(ctypes.c_float),  # sums_quant
    ctypes.POINTER(ctypes.c_float),  # sums_topk
    ctypes.c_int,  # seq_len
    ctypes.c_int,  # head_num
    ctypes.c_int,  # batch_size
    ctypes.c_int,  # head_offset
    ctypes.c_int,  # batch_offset
    ctypes.c_int,  # thread_num
]
lib.prepare_softmax.restype = None
lib.set_ready_flag.argtypes = []
lib.set_ready_flag.restype = None
lib.is_finished.argtypes = []
lib.is_finished.restype = ctypes.c_long
lib.clear_flags.argtypes = []
lib.clear_flags.restype = None
lib.get_duration.argtypes = []
lib.get_duration.restype = ctypes.c_double
lib.wait_finished.argtypes = []
lib.wait_finished.restype = ctypes.c_long


def test_gemv(
    total_work,
    batch_size,
    S_len,
    thread_num,
    Dh,
    q_head_num,
    kv_head_num,
    topk_indices,
    topk_num,
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
    # Tasks for each thread
    def task_value_gemv():
        # Elevate the process to the highest priority(real-time class)
        sched_param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, sched_param)
        except PermissionError:
            print("Permission denied. Try running as root.")

        # os.sched_setaffinity(0, {5})
        lib.prepare_value_gemv(
            # ctypes.cast(values_keys.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            # ctypes.cast(logits_queries.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            # ctypes.cast(result_logits.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            values_keys.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            logits_queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            result_logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(q_head_num),
            ctypes.c_int(kv_head_num),
            ctypes.c_int(batch_size),
            ctypes.c_int(S_len),
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

        # os.sched_setaffinity(0, {5})
        lib.prepare_key_gemv(
            ctypes.cast(values_keys.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(logits_queries.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(result_logits.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(q_head_num),
            ctypes.c_int(kv_head_num),
            ctypes.c_int(batch_size),
            ctypes.c_int(S_len),
            ctypes.c_int(Dh),
            ctypes.c_int(kv_head_offset),
            ctypes.c_int(kv_batch_offset),
            ctypes.c_int(logits_queries_head_offset),
            ctypes.c_int(logits_queries_batch_offset),
            ctypes.c_int(out_logits_head_offset),
            ctypes.c_int(out_logits_batch_offset),
            ctypes.c_int(thread_num),
        )

    def task_value_gemv_half():
        # Elevate the process to the highest priority(real-time class)
        sched_param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, sched_param)
        except PermissionError:
            print("Permission denied. Try running as root.")

        # os.sched_setaffinity(0, {5})
        if values_keys.dtype == np.float16:
            lib.prepare_value_gemv_half(
                values_keys.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                logits_queries.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                result_logits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                topk_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int(topk_num),
                ctypes.c_int(q_head_num),
                ctypes.c_int(kv_head_num),
                ctypes.c_int(batch_size),
                ctypes.c_int(S_len),
                ctypes.c_int(Dh),
                ctypes.c_int(kv_head_offset),
                ctypes.c_int(kv_batch_offset),
                ctypes.c_int(logits_queries_head_offset),
                ctypes.c_int(logits_queries_batch_offset),
                ctypes.c_int(out_logits_head_offset),
                ctypes.c_int(out_logits_batch_offset),
                ctypes.c_int(thread_num),
                ctypes.c_int(total_work),
            )
        else:
            lib.prepare_value_gemv_half(
                ctypes.cast(values_keys.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
                ctypes.cast(logits_queries.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
                ctypes.cast(result_logits.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
                ctypes.cast(topk_indices.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int(topk_num),
                ctypes.c_int(q_head_num),
                ctypes.c_int(kv_head_num),
                ctypes.c_int(batch_size),
                ctypes.c_int(S_len),
                ctypes.c_int(Dh),
                ctypes.c_int(kv_head_offset),
                ctypes.c_int(kv_batch_offset),
                ctypes.c_int(logits_queries_head_offset),
                ctypes.c_int(logits_queries_batch_offset),
                ctypes.c_int(out_logits_head_offset),
                ctypes.c_int(out_logits_batch_offset),
                ctypes.c_int(thread_num),
                ctypes.c_int(total_work),
            )

    def task_key_gemv_half():
        # Elevate the process to the highest priority(real-time class)
        sched_param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, sched_param)
        except PermissionError:
            print("Permission denied. Try running as root.")

        # os.sched_setaffinity(0, {71})
        if values_keys.dtype == np.float16:
            lib.prepare_key_gemv_half(
                values_keys.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                logits_queries.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                result_logits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                topk_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int(topk_num),
                ctypes.c_int(q_head_num),
                ctypes.c_int(kv_head_num),
                ctypes.c_int(batch_size),
                ctypes.c_int(S_len),
                ctypes.c_int(Dh),
                ctypes.c_int(kv_head_offset),
                ctypes.c_int(kv_batch_offset),
                ctypes.c_int(logits_queries_head_offset),
                ctypes.c_int(logits_queries_batch_offset),
                ctypes.c_int(out_logits_head_offset),
                ctypes.c_int(out_logits_batch_offset),
                ctypes.c_int(thread_num),
                ctypes.c_int(total_work),
            )
        else:
            lib.prepare_key_gemv_half(
                ctypes.cast(values_keys.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
                ctypes.cast(logits_queries.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
                ctypes.cast(result_logits.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
                ctypes.cast(topk_indices.data_ptr(), ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_int(topk_num),
                ctypes.c_int(q_head_num),
                ctypes.c_int(kv_head_num),
                ctypes.c_int(batch_size),
                ctypes.c_int(S_len),
                ctypes.c_int(Dh),
                ctypes.c_int(kv_head_offset),
                ctypes.c_int(kv_batch_offset),
                ctypes.c_int(logits_queries_head_offset),
                ctypes.c_int(logits_queries_batch_offset),
                ctypes.c_int(out_logits_head_offset),
                ctypes.c_int(out_logits_batch_offset),
                ctypes.c_int(thread_num),
                ctypes.c_int(total_work),
            )

    ######
    # Create and run a thread
    if is_key_gemv:
        if values_keys.dtype == torch.float:
            thread = threading.Thread(target=task_key_gemv)
        else:
            thread = threading.Thread(target=task_key_gemv_half)
    else:
        if values_keys.dtype == torch.float:
            thread = threading.Thread(target=task_value_gemv)
        else:
            thread = threading.Thread(target=task_value_gemv_half)
    thread.start()
    ######

    time.sleep(5)

    start_t = time.perf_counter_ns()
    lib.set_ready_flag()
    duration = lib.wait_finished()
    end_t = time.perf_counter_ns()

    lib.clear_flags()
    thread.join()
    # return end_t - start_t
    return duration


def aligned_array(shape, dtype, alignment=64):
    """Create a memory-aligned numpy array."""
    nbytes = np.prod(shape) * np.dtype(dtype).itemsize
    buffer = np.empty(nbytes + alignment, dtype=np.uint8)
    start_index = -buffer.ctypes.data % alignment
    aligned_buffer = buffer[start_index : start_index + nbytes]
    return np.ndarray(shape, dtype=dtype, buffer=aligned_buffer)


def test_value_gemv(batch_size, S_len, ratio, thread_num, is_gqa, dtype=np.float16):
    # Elevate the process to the highest priority(real-time class)
    sched_param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, sched_param)
    except PermissionError:
        print("Permission denied. Try running as root.")

    q_head_num = 32
    kv_head_num = 8 if is_gqa else 32
    Dh = 128
    topk_num = round(S_len * ratio)

    print("Value GEMV FP16")
    print(
        f"BS: {batch_size}, S: {S_len}, Top-k num: {topk_num}, Dh: {Dh}, q_head_num: {q_head_num}, kv_head_num: {kv_head_num}, thread_num: {thread_num}"
    )

    # Allocate memory
    (
        values,
        logits,
        outs,
        topk_indices,
        kv_batch_offset,
        kv_head_offset,
        logits_queries_batch_offset,
        logits_queries_head_offset,
        out_logits_batch_offset,
        out_logits_head_offset,
        total_work,
    ) = layout_1_value(
        topk_num,
        q_head_num,
        kv_head_num,
        batch_size,
        S_len,
        Dh,
        dtype,
    )

    # [x] Correctness Test
    # logits_gpu = (
    #     logits.to(device="cuda")
    #     .transpose(1, 2)
    #     .reshape(batch_size * q_head_num, 1, topk_num)
    # )
    # values_gpu = values.permute(0, 2, 1, 3).to(
    #     device="cuda"
    # )  # (batch_size, kv_head_num, seq_len, Dh)

    total_sec = 0
    for _ in range(ITER):
        nano_sec = test_gemv(
            total_work,
            batch_size,
            S_len,
            thread_num,
            Dh,
            q_head_num,
            kv_head_num,
            topk_indices,
            topk_num,
            values,
            logits,
            outs,
            kv_head_offset,
            kv_batch_offset,
            logits_queries_head_offset,
            logits_queries_batch_offset,
            out_logits_head_offset,
            out_logits_batch_offset,
            False,
        )
        ##########################################
        # Performance and Correctness
        # [x] Performance
        print("=====================================================")
        if isinstance(values, torch.Tensor):
            # DEBUG
            print(f"Value element size: {values.element_size()}")
            print(f"Logits element size: {logits.element_size()}")
            total_bytes = (
                values.element_size() * kv_head_num * batch_size * topk_num * Dh
                + logits.element_size() * q_head_num * batch_size * topk_num
            )
        else:
            # DEBUG
            print(f"Value element size: {values.itemsize}")
            print(f"Logits element size: {logits.itemsize}")
            total_bytes = (
                kv_head_num * batch_size * topk_num * Dh * values.itemsize
                + q_head_num * batch_size * topk_num * logits.itemsize,
            )

        micro_sec = nano_sec / 1e3
        sec = nano_sec / 1e9
        throughput = (total_bytes / 1e9) / sec

        print(f"Took {micro_sec} microseconds")
        print(f"Total bytes: {total_bytes / 1e9} GB")
        print(f"Throughput(GB/s): {throughput}")
        ######
        # [x] Correctness Test
        # print(outs)
        # topk_indices_gpu = (
        #     topk_indices.unsqueeze(-1)
        #     .expand(-1, -1, -1, Dh)
        #     .to(device="cuda", dtype=torch.int64)
        # )  # (batch_size, q_head_num, topk_num, Dh)

        # values_selected = torch.gather(
        #     values_gpu, dim=2, index=topk_indices_gpu
        # )  # (batch_size, kv_head_num, topk_num, Dh)
        # values_selected = values_selected.repeat_interleave(
        #     4, dim=1
        # )  # (batch_size, q_head_num, topk_num, Dh)
        # values_selected = values_selected.reshape(batch_size * q_head_num, topk_num, Dh)

        # """
        # logits_selected: (batch_size * q_head_num, 1, K)
        # values_selected: (batch_size * q_head_num, K, Dh)
        # """
        # attn_out = torch.bmm(logits_gpu, values_selected).reshape(
        #     batch_size, q_head_num, Dh
        # )
        # print(f"attn_out: {attn_out}")
        # print(f"outs: {outs}")
        # mse = torch.mean((outs - attn_out.to(device="cpu")) ** 2)
        # mae = torch.mean(abs(outs - attn_out.to(device="cpu")))

        # del values_selected
        # torch.cuda.empty_cache()
        # print(f"MSE: {mse}")
        # print(f"MAE: {mae}")
        ######
        ##########################################

        # Change top-k indices
        if dtype == np.float16:
            topk_indices = np.random.randint(0, S_len, size=topk_num, dtype=np.uint16)
        else:
            topk_indices_shape = (batch_size, kv_head_num, topk_num)
            # Populate the tensor
            for i in range(topk_indices_shape[0]):  # Iterate over the first dimension
                for j in range(
                    topk_indices_shape[1]
                ):  # Iterate over the second dimension
                    # [x] High Sparsity
                    # topk_indices[i, j] = torch.randperm(S_len, dtype=torch.int16)[
                    #     : topk_indices_shape[2]
                    # ]
                    # [x] Low Sparsity
                    topk_indices[i, j], _ = torch.sort(
                        torch.randperm(S_len, dtype=torch.int16)[
                            : topk_indices_shape[2]
                        ]
                    )  # Cast randperm to int16
            # [x] Combine 8 tokens => No effect
            # topk_indices_shape = (batch_size, kv_head_num, topk_num)
            # topk_indices = torch.empty(topk_indices_shape, dtype=torch.int16)
            # suc = 8
            # # Populate the tensor
            # div = int(topk_num / suc)

            # gap = int(S_len / div)
            # lower_bound = 0
            # upper_bound = lower_bound + gap

            # for i in range(topk_indices_shape[0]):  # Iterate over the first dimension
            #     for j in range(
            #         topk_indices_shape[1]
            #     ):  # Iterate over the second dimension
            #         indices = []
            #         remains = topk_num % suc
            #         lower_bound = 0
            #         upper_bound = lower_bound + gap
            #         for _ in range(div):
            #             num = torch.randint(lower_bound, upper_bound - suc, (1,)).item()
            #             for s in range(suc):
            #                 indices.append(num + s)
            #             lower_bound = upper_bound
            #             upper_bound += gap
            #         while remains:
            #             abc = torch.randint(S_len, (1,)).item()
            #             if abc in indices:
            #                 continue
            #             indices.append(abc)
            #             remains -= 1

            #         assert len(indices) == topk_indices_shape[2], "Indices Error!!!"
            #         topk_indices[i, j] = torch.tensor(indices)

        # (
        #     values,
        #     logits,
        #     outs,
        #     topk_indices,
        #     kv_batch_offset,
        #     kv_head_offset,
        #     logits_queries_batch_offset,
        #     logits_queries_head_offset,
        #     out_logits_batch_offset,
        #     out_logits_head_offset,
        # ) = layout_3_value(
        #     topk_num,
        #     q_head_num,
        #     kv_head_num,
        #     batch_size,
        #     S_len,
        #     Dh,
        #     dtype,
        # )
        # OS-specific cache flush (Linux example)
        if os.name == "posix":
            os.system(
                "sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1"
            )
        total_sec = total_sec + nano_sec
    total_sec = total_sec / ITER
    ##########################################
    # Performance and Correctness
    # [x] Performance
    print("=====================================================")
    if isinstance(values, torch.Tensor):
        # DEBUG
        # print(f"Number of elements in Keys: {values.element_size()}")
        # print(f"Number of elements in Logits: {logits.element_size()}")
        # print(f"Keys element size: {values.element_size()}")
        # print(f"Logits element size: {logits.element_size()}")
        # total_bytes = (
        #     values.element_size() * values.nelement()
        #     + logits.element_size() * logits.nelement()
        # )
        total_bytes = (
            values.element_size() * kv_head_num * batch_size * topk_num * Dh
            + logits.element_size() * q_head_num * batch_size * topk_num
        )
    else:
        # DEBUG
        # print(f"Number of elements in Keys: {values.size}")
        # print(f"Number of elements in Logits: {logits.size}")
        # print(f"Keys element size: {values.itemsize}")
        # print(f"Logits element size: {logits.itemsize}")
        total_bytes = (
            kv_head_num * batch_size * topk_num * Dh * values.itemsize
            + q_head_num * batch_size * topk_num * logits.itemsize
        )

    micro_sec = total_sec / 1e3
    sec = total_sec / 1e9
    throughput = (total_bytes / 1e9) / sec

    print(f"{micro_sec:.2f} microseconds")
    # print(f"Total bytes: {total_bytes / 1e9} GB")
    print(f"{throughput:.2f} GB/s")
    ######

    # [x] Check the correctness
    # values = values.transpose(1, 2)
    # logits = logits.to("cuda")
    # values = values.to("cuda")
    # attn_weights = _attention_weights(logits, values, None, batch_size, K, head_num)
    # attn_weights = attn_weights.reshape(head_num, batch_size, K)
    # outs = outs.reshape(head_num, batch_size, K)

    # mse = torch.mean((outs - attn_weights.to(device="cpu")) ** 2)
    # mae = torch.mean(abs(outs - attn_weights.to(device="cpu")))
    # print(f"MSE: {mse}")
    # print(f"MAE: {mae}")
    ######
    ##########################################


def _attention_weights(q, k, mask, b, src_s, n_head):
    # shape: (b * n_head, 1, s)
    attn_weights = torch.bmm(q, k)
    # # shape: (b, 1, 1, s)
    # if mask is not None:
    #     mask = mask.view(b, 1, 1, src_s)
    # # shape: (b * n_head, 1, s)
    # attn_weights = attn_weights.view(b, n_head, 1, src_s)
    # if mask is not None:
    #     attn_weights = torch.where(mask, attn_weights, -1e4)

    # attn_weights = attn_weights.view(b * n_head, 1, src_s)
    # attn_weights = F.softmax(attn_weights, dim=2)

    return attn_weights


def test_key_gemv(batch_size, S_len, ratio, thread_num, is_gqa, dtype=torch.float16):
    # Elevate the process to the highest priority(real-time class)
    sched_param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, sched_param)
    except PermissionError:
        print("Permission denied. Try running as root.")

    q_head_num = 4
    kv_head_num = 1 if is_gqa else 4
    Dh = 128
    topk_num = round(S_len * ratio)
    print(f"Key GEMV FP16")
    print(
        f"BS: {batch_size}, S: {S_len}, Top-k ratio: {ratio}, Dh: {Dh}, q_head_num: {q_head_num}, kv_head_num: {kv_head_num}, thread_num: {thread_num}"
    )

    ############################################
    # Allocate memory
    """
    K: (B, H_kv, S, Dh)
    Q: (B, H_q, Dh)
    L: (B, H_q, K)
    """
    (
        keys,
        queries,
        logits,
        topk_indices,
        kv_batch_offset,
        kv_head_offset,
        logits_queries_head_offset,
        logits_queries_batch_offset,
        out_logits_head_offset,
        out_logits_batch_offset,
        total_work,
    ) = layout_1_key(
        topk_num,
        q_head_num,
        kv_head_num,
        batch_size,
        S_len,
        Dh,
        dtype,
    )
    ############################################

    total_sec = 0
    for _ in range(ITER):
        nano_sec = test_gemv(
            total_work,
            batch_size,
            S_len,
            thread_num,
            Dh,
            q_head_num,
            kv_head_num,
            topk_indices,
            topk_num,
            keys,
            queries,
            logits,
            kv_head_offset,
            kv_batch_offset,
            logits_queries_head_offset,
            logits_queries_batch_offset,
            out_logits_head_offset,
            out_logits_batch_offset,
            True,
        )
        total_sec = total_sec + nano_sec
        ##########################################
        # [x] Performance
        print("=====================================================")

        if isinstance(keys, torch.Tensor):
            # DEBUG
            print(f"Keys element size: {keys.element_size()}")
            print(f"queires element size: {queries.element_size()}")
            total_bytes = (
                keys.element_size() * batch_size * kv_head_num * topk_num * Dh
                + queries.element_size() * batch_size * q_head_num * Dh
            )
        else:
            # DEBUG
            print(f"Keys element size: {keys.itemsize}")
            print(f"queires element size: {queries.itemsize}")
            total_bytes = (
                batch_size * kv_head_num * topk_num * Dh * keys.itemsize
                + batch_size * q_head_num * Dh * queries.itemsize,
            )

        micro_sec = nano_sec / 1e3
        sec = nano_sec / 1e9
        throughput = (total_bytes / 1e9) / sec

        print(f"Took {micro_sec} microseconds")
        print(f"Total bytes: {total_bytes / 1e9} GB")
        print(f"Throughput(GB/s): {throughput}")
        ######
        total_sec = total_sec + nano_sec
        ######

        # # [x] Check the correctness
        # keys = keys.transpose(1, 2)
        # queries = queries.to("cuda")
        # keys = keys.to("cuda")
        # attn_weights = _attention_weights(queries, keys, None, batch_size, K, head_num)
        # attn_weights = attn_weights.reshape(head_num, batch_size, K)
        # logits = logits.reshape(head_num, batch_size, K)

        # mse = torch.mean((logits - attn_weights.to(device="cpu")) ** 2)
        # mae = torch.mean(abs(logits - attn_weights.to(device="cpu")))
        # # print(f"MSE: {mse}")
        # # print(f"MAE: {mae}")
        # ######
        ##########################################

        # if dtype == np.float16:
        #     # Numpy version
        #     keys = aligned_array(
        #         (kv_head_num, batch_size, K, Dh), dtype=np.float16, alignment=64
        #     )
        #     queries = aligned_array(
        #         (q_head_num, batch_size, Dh), dtype=np.float16, alignment=64
        #     )
        #     logits = aligned_array(
        #         (q_head_num, batch_size, K), dtype=np.float16, alignment=64
        #     )
        #     keys[:] = np.random.rand(*keys.shape).astype(np.float16)
        #     queries[:] = np.random.rand(*queries.shape).astype(np.float16)
        #     # Ensure alignment (numpy arrays are typically well-aligned for SIMD operations)
        #     assert keys.ctypes.data % 64 == 0, "keys is not 64-byte aligned!"
        #     assert queries.ctypes.data % 64 == 0, "queries is not 64-byte aligned!"
        #     assert logits.ctypes.data % 64 == 0, "logits is not 64-byte aligned!"
        # else:
        #     # Torch version
        #     keys = torch.rand(kv_head_num * batch_size, K, Dh, dtype=dtype)
        #     queries = torch.rand(q_head_num * batch_size, 1, Dh, dtype=dtype)
        #     logits = torch.zeros(q_head_num * batch_size, 1, K, dtype=dtype)
        #     assert keys.data_ptr() % 64 == 0, "keys is not 64-byte aligned!"
        #     assert queries.data_ptr() % 64 == 0, "queries is not 64-byte aligned!"
        #     assert logits.data_ptr() % 64 == 0, "logits is not 64-byte aligned!"
        topk_indices_shape = (batch_size, kv_head_num, topk_num)
        topk_indices = torch.empty(topk_indices_shape, dtype=torch.int16)
        # Populate the tensor
        for i in range(topk_indices_shape[0]):  # Iterate over the first dimension
            for j in range(topk_indices_shape[1]):  # Iterate over the second dimension
                # [x] For high sparsity
                # topk_indices[i, j] = torch.randperm(S_len, dtype=torch.int16)[
                #     : topk_indices_shape[2]
                # ]  # Cast randperm to int16
                # [x] For low sparsity
                topk_indices[i, j], _ = torch.sort(
                    torch.randperm(S_len, dtype=torch.int16)[: topk_indices_shape[2]]
                )  # Cast randperm to int16

        # OS-specific cache flush (Linux example)
        if os.name == "posix":
            os.system(
                "sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1"
            )
    total_sec = total_sec / ITER
    ##########################################
    # Performance and Correctness
    # [x] Performance
    print("=====================================================")
    if isinstance(keys, torch.Tensor):
        # DEBUG
        # print(f"Number of elements in Keys: {keys.element_size()}")
        # print(f"Number of elements in Logits: {queries.element_size()}")
        # print(f"Keys element size: {keys.element_size()}")
        # print(f"Logits element size: {queries.element_size()}")
        total_bytes = (
            keys.element_size() * batch_size * kv_head_num * topk_num * Dh
            + queries.element_size() * batch_size * q_head_num * Dh
        )
    else:
        # DEBUG
        # print(f"Number of elements in Keys: {keys.size}")
        # print(f"Number of elements in Logits: {queries.size}")
        # print(f"Keys element size: {keys.itemsize}")
        # print(f"Logits element size: {queries.itemsize}")
        total_bytes = keys.size * keys.itemsize + queries.size * queries.itemsize

    micro_sec = total_sec / 1e3
    sec = total_sec / 1e9
    throughput = (total_bytes / 1e9) / sec

    print(f"{micro_sec:.2f} microseconds")
    # print(f"Total bytes: {total_bytes / 1e9} GB")
    print(f"{throughput:.2f} GB/s")
    ######

    # [x] Check the correctness
    # keys = keys.transpose(1, 2)
    # queries = queries.to("cuda")
    # keys = keys.to("cuda")
    # attn_weights = _attention_weights(queries, keys, None, batch_size, K, head_num)
    # attn_weights = attn_weights.reshape(head_num, batch_size, K)
    # logits = logits.reshape(head_num, batch_size, K)

    # mse = torch.mean((logits - attn_weights.to(device="cpu")) ** 2)
    # mae = torch.mean(abs(logits - attn_weights.to(device="cpu")))
    # print(f"MSE: {mse}")
    # print(f"MAE: {mae}")
    ######
    ##########################################


###############################################
# Test for Softmax
def test_softmax_threads(
    qk,
    max_values,
    sums_quant,
    sums_topk,
    head_num,
    batch_size,
    seq_len,
    head_offset,
    batch_offset,
    thread_num,
):
    ##############################
    # Tasks for each thread
    def task_softmax():
        # Elevate the process to the highest priority(real-time class)
        sched_param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, sched_param)
        except PermissionError:
            print("Permission denied. Try running as root.")

        # os.sched_setaffinity(0, {5})
        lib.prepare_softmax(
            ctypes.cast(qk.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(max_values.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(sums_quant.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(sums_topk.data_ptr(), ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(seq_len),
            ctypes.c_int(head_num),
            ctypes.c_int(batch_size),
            ctypes.c_int(head_offset),
            ctypes.c_int(batch_offset),
            ctypes.c_int(thread_num),
        )

    # def task_softmax_half():
    #     # Elevate the process to the highest priority(real-time class)
    #     sched_param = os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO))
    #     try:
    #         os.sched_setscheduler(0, os.SCHED_FIFO, sched_param)
    #     except PermissionError:
    #         print("Permission denied. Try running as root.")

    #     # void prepare_softmax(float *qk, const float *max_arr, const size_t seq_len,
    #     #              const size_t head_num, const size_t batch_size,
    #     #              const size_t head_offset, const size_t batch_offset,
    #     #              const int thread_idx, const int thread_num)
    #     os.sched_setaffinity(0, {5})
    #     lib.prepare_softmax(
    #         ctypes.cast(qk.data_ptr(), ctypes.POINTER(ctypes.c_float)),
    #         ctypes.cast(max_values.data_ptr(), ctypes.POINTER(ctypes.c_float)),
    #         ctypes.c_int(seq_len),
    #         ctypes.c_int(head_num),
    #         ctypes.c_int(batch_size),
    #         ctypes.c_int(head_offset),
    #         ctypes.c_int(batch_offset),
    #         ctypes.c_int(thread_num),
    #     )

    ##############################

    # Create and run a thread
    if qk.dtype == torch.float:
        thread = threading.Thread(target=task_softmax)
    else:
        pass
    # thread = threading.Thread(target=task_softmax_half)

    thread.start()
    ######

    time.sleep(1)

    print("=====================================================")
    start_t = time.perf_counter_ns()
    lib.set_ready_flag()
    lib.wait_finished()
    end_t = time.perf_counter_ns()

    duration = (end_t - start_t) / 1e3
    print(f"Took {duration} microseconds")
    lib.clear_flags()
    thread.join()


#  ./main 64 1024 81 8 => 500 ms
def test_softmax(batch_size, K, thread_num, dtype=torch.float):
    head_num = 32
    seq_len = 1024

    # batch_size = 1

    head_offset = batch_size * K
    batch_offset = K

    qk = torch.rand(head_num, batch_size, seq_len, dtype=dtype)

    # for _ in range(ITER):
    # [x] Produce the answer by Pytorch
    start_torch = time.perf_counter_ns()
    logits_from_torch = F.softmax(qk, dim=-1)
    end_torch = time.perf_counter_ns()

    duration = (end_torch - start_torch) / 1e3
    # print(f"Took {duration} microseconds")

    # [ ] Select top-k indices
    true_logits, topk_indices = logits_from_torch.topk(K, dim=-1, sorted=False)
    qk_topk = torch.gather(qk, dim=-1, index=topk_indices)

    # [x] Calculate max_values and sums_quant before Softmax
    max_values, max_indices = torch.max(qk, dim=-1, keepdim=True)
    qk_exp = torch.exp(qk - max_values)
    sums_quant = torch.sum(qk_exp, dim=-1, keepdim=True)
    logits_tmp = qk_exp / sums_quant
    logits_tmp_topk, _ = logits_tmp.topk(K, dim=-1, sorted=False)

    sums_topk = torch.empty_like(sums_quant)
    # DEBUG
    # print(f"qk_exp: {qk_exp.shape}")
    # print(qk_exp[0])
    # print(f"sums_quant: {sums_quant.shape}")
    # print(sums_quant)

    # [x] Run CPU Softmax
    test_softmax_threads(
        qk_topk,
        max_values,
        sums_quant,
        sums_topk,
        head_num,
        batch_size,
        K,
        head_offset,
        batch_offset,
        thread_num,
    )
    # DEBUG
    print(f"logits_tmp_topk: {logits_tmp_topk.shape}")
    print(logits_tmp_topk[0])
    print(f"true_logits: {true_logits.shape}")
    print(true_logits[0])
    print(f"qk_topk: {qk_topk.shape}")
    print(qk_topk[0])

    # [x] Calculate the MSE and MAE
    # # Calculate MSE
    # mse = torch.mean((logits_from_torch - qk) ** 2)
    # print("Mean Squared Error (MSE):", mse.item())

    # # Calculate MAE
    # mae = torch.mean(torch.abs(logits_from_torch - qk))
    # print("Mean Absolute Error (MAE):", mae.item())


###############################################


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
    parser.add_argument("--S_len", type=int, default=4096, help="Length of a sequence")
    parser.add_argument("--ratio", type=float, default=0.01, help="Top-k ratio")
    parser.add_argument("--thread_num", type=int, default=48, help="Number of threads")
    parser.add_argument("--numpy", action="store_true", help="Numpy or Torch")
    parser.add_argument("--key_gemv", action="store_true", help="Test Key GEMV")
    parser.add_argument("--softmax", action="store_true", help="Test Softmax")
    parser.add_argument("--gqa", action="store_true", help="GQA or MHA")

    args = parser.parse_args()

    if args.softmax:
        test_softmax(args.batch_size, args.K, args.thread_num, dtype=torch.float)
    else:
        if args.key_gemv:
            if args.numpy:
                test_key_gemv(
                    args.batch_size,
                    args.S_len,
                    args.ratio,
                    args.thread_num,
                    args.gqa,
                    dtype=np.float16,
                )
            else:
                test_key_gemv(
                    args.batch_size,
                    args.S_len,
                    args.ratio,
                    args.thread_num,
                    args.gqa,
                    dtype=torch.float16,
                )
        else:
            if args.numpy:
                test_value_gemv(
                    args.batch_size,
                    args.S_len,
                    args.ratio,
                    args.thread_num,
                    args.gqa,
                    dtype=np.float16,
                )
            else:
                test_value_gemv(
                    args.batch_size,
                    args.S_len,
                    args.ratio,
                    args.thread_num,
                    args.gqa,
                    dtype=torch.float16,
                )


if __name__ == "__main__":
    # os.sched_setaffinity(0, {71})
    main()
