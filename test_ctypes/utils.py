import numpy as np
import ctypes
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


# (batch, head, S_len, Dh) version
def layout_1_value(
    topk_num,
    q_head_num,
    kv_head_num,
    batch_size,
    S_len,
    Dh,
    dtype=torch.float16,
):
    kv_batch_offset = kv_head_num * S_len * Dh
    kv_head_offset = S_len * Dh
    logits_queries_batch_offset = q_head_num * topk_num
    logits_queries_head_offset = topk_num
    out_logits_batch_offset = q_head_num * Dh
    out_logits_head_offset = Dh

    if dtype == np.float16:
        topk_indices = np.random.randint(0, S_len, size=topk_num, dtype=np.uint16)
        # Memory-aligned allocation using NumPy (default alignment is sufficient for most cases)
        values = aligned_array(
            (batch_size, kv_head_num, S_len, Dh), dtype=np.float16, alignment=64
        )
        logits = aligned_array(
            (batch_size, q_head_num, topk_num), dtype=np.float16, alignment=64
        )
        results = aligned_array(
            (batch_size, q_head_num, Dh), dtype=np.float16, alignment=64
        )
        # Fill values and logits with random values
        values[:] = np.random.rand(*values.shape).astype(np.float16)
        logits[:] = np.random.rand(*logits.shape).astype(np.float16)

        # Ensure alignment (numpy arrays are typically well-aligned for SIMD operations)
        assert values.ctypes.data % 64 == 0, "values is not 64-byte aligned!"
        assert logits.ctypes.data % 64 == 0, "logits is not 64-byte aligned!"
        assert results.ctypes.data % 64 == 0, "results is not 64-byte aligned!"
    else:
        topk_indices, _ = torch.sort(
            torch.randint(0, S_len - 1, (topk_num,), dtype=torch.int16)
        )
        # Memory-aligned allocation using Torch
        values = torch.rand(batch_size, kv_head_num, S_len, Dh, dtype=dtype)
        logits = torch.rand(batch_size, q_head_num, topk_num, dtype=dtype)
        results = torch.zeros(batch_size, q_head_num, Dh, dtype=dtype)

        # Check memory alignment
        assert values.data_ptr() % 64 == 0, "values is not 64-byte aligned!"
        assert logits.data_ptr() % 64 == 0, "logits is not 64-byte aligned!"
        assert results.data_ptr() % 64 == 0, "results is not 64-byte aligned!"
    return (
        values,
        logits,
        results,
        topk_indices,
        kv_batch_offset,
        kv_head_offset,
        logits_queries_batch_offset,
        logits_queries_head_offset,
        out_logits_batch_offset,
        out_logits_head_offset,
    )


def layout_2_value(
    topk_num,
    q_head_num,
    kv_head_num,
    batch_size,
    S_len,
    Dh,
    dtype=torch.float16,
):
    # Result of CPU Key GEMV
    kv_batch_offset = S_len * kv_head_num * Dh
    kv_head_offset = kv_head_num * Dh
    logits_queries_batch_offset = topk_num * q_head_num
    logits_queries_head_offset = q_head_num
    out_logits_batch_offset = q_head_num * Dh
    out_logits_head_offset = Dh

    if dtype == np.float16:
        topk_indices = np.random.randint(0, S_len, size=topk_num, dtype=np.uint16)
        # Memory-aligned allocation using NumPy (default alignment is sufficient for most cases)
        values = aligned_array(
            (batch_size, S_len, kv_head_num, Dh), dtype=np.float16, alignment=64
        )
        logits = aligned_array(
            (batch_size, topk_num, q_head_num), dtype=np.float16, alignment=64
        )
        results = aligned_array(
            (batch_size, q_head_num, Dh), dtype=np.float16, alignment=64
        )
        # Fill values and logits with random values
        values[:] = np.random.rand(*values.shape).astype(np.float16)
        logits[:] = np.random.rand(*logits.shape).astype(np.float16)

        # Ensure alignment (numpy arrays are typically well-aligned for SIMD operations)
        assert values.ctypes.data % 64 == 0, "values is not 64-byte aligned!"
        assert logits.ctypes.data % 64 == 0, "logits is not 64-byte aligned!"
        assert results.ctypes.data % 64 == 0, "results is not 64-byte aligned!"
    else:
        topk_indices, _ = torch.sort(
            torch.randint(0, S_len - 1, (topk_num,), dtype=torch.int16)
        )
        # Memory-aligned allocation using Torch
        values = torch.rand(batch_size, S_len, kv_head_num, Dh, dtype=dtype)
        logits = torch.rand(batch_size, topk_num, q_head_num, dtype=dtype)
        results = torch.zeros(batch_size, q_head_num, Dh, dtype=dtype)

        # Check memory alignment
        assert values.data_ptr() % 64 == 0, "values is not 64-byte aligned!"
        assert logits.data_ptr() % 64 == 0, "logits is not 64-byte aligned!"
        assert results.data_ptr() % 64 == 0, "results is not 64-byte aligned!"
    return (
        values,
        logits,
        results,
        topk_indices,
        kv_batch_offset,
        kv_head_offset,
        logits_queries_batch_offset,
        logits_queries_head_offset,
        out_logits_batch_offset,
        out_logits_head_offset,
    )


def layout_3_value(
    topk_num,
    q_head_num,
    kv_head_num,
    batch_size,
    S_len,
    Dh,
    dtype=torch.float16,
):
    # Result of CPU Key GEMV
    kv_batch_offset = batch_size * kv_head_num * Dh
    kv_head_offset = kv_head_num * Dh
    logits_queries_batch_offset = batch_size * q_head_num
    logits_queries_head_offset = q_head_num
    out_logits_batch_offset = q_head_num * Dh
    out_logits_head_offset = Dh

    if dtype == np.float16:
        topk_indices = np.random.randint(0, S_len, size=topk_num, dtype=np.uint16)
        # Memory-aligned allocation using NumPy (default alignment is sufficient for most cases)
        values = aligned_array(
            (S_len, batch_size, kv_head_num, Dh), dtype=np.float16, alignment=64
        )
        logits = aligned_array(
            (topk_num, batch_size, q_head_num), dtype=np.float16, alignment=64
        )
        results = aligned_array(
            (batch_size, q_head_num, Dh), dtype=np.float16, alignment=64
        )
        # Fill values and logits with random values
        values[:] = np.random.rand(*values.shape).astype(np.float16)
        logits[:] = np.random.rand(*logits.shape).astype(np.float16)

        # Ensure alignment (numpy arrays are typically well-aligned for SIMD operations)
        assert values.ctypes.data % 64 == 0, "values is not 64-byte aligned!"
        assert logits.ctypes.data % 64 == 0, "logits is not 64-byte aligned!"
        assert results.ctypes.data % 64 == 0, "results is not 64-byte aligned!"
    else:
        topk_indices, _ = torch.sort(
            torch.randint(0, S_len - 1, (topk_num,), dtype=torch.int16)
        )
        # Memory-aligned allocation using Torch
        values = torch.rand(S_len, batch_size, kv_head_num, Dh, dtype=dtype)
        logits = torch.rand(S_len, batch_size, q_head_num, dtype=dtype)
        results = torch.zeros(batch_size, q_head_num, Dh, dtype=dtype)

        # Check memory alignment
        assert values.data_ptr() % 64 == 0, "values is not 64-byte aligned!"
        assert logits.data_ptr() % 64 == 0, "logits is not 64-byte aligned!"
        assert results.data_ptr() % 64 == 0, "results is not 64-byte aligned!"
    return (
        values,
        logits,
        results,
        topk_indices,
        kv_batch_offset,
        kv_head_offset,
        logits_queries_batch_offset,
        logits_queries_head_offset,
        out_logits_batch_offset,
        out_logits_head_offset,
    )
