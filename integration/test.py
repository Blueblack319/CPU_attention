import numpy as np
import build.attn_module as attn_module
import ctypes


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


def test_single_process():
    # Initialize example data
    batch_size = 32
    K = 81
    thread_num = 4
    Dh = 128
    head_num = 32

    # values = np.random.rand(head_num * batch_size * K * Dh).astype(np.float32, order='C', align=True)
    # logits = np.random.rand(head_num * batch_size * K).astype(np.float32, order='C', align=True)
    # result = np.zeros(head_num * batch_size * Dh, dtype=np.float32, order='C', align=True)

    values = aligned_array(size=head_num * batch_size * K * Dh)
    logits = aligned_array(size=head_num * batch_size * K)
    result = aligned_array(size=head_num * batch_size * Dh)
    values[:] = np.random.rand(head_num * batch_size * K * Dh)
    logits[:] = np.random.rand(head_num * batch_size * K)
    result[:] = np.zeros(head_num * batch_size * Dh)

    # values = np.random.rand((head_num * batch_size, K * Dh)).astype(np.float32, order='C')
    # logits = np.random.rand((head_num * batch_size, K)).astype(np.float32, order='C')
    # result = np.zeros((head_num * batch_size, Dh), dtype=np.float32, order='C')

    total_work = head_num * batch_size
    work_per_thread = int(total_work / thread_num)

    kv_head_offset = batch_size * K * Dh
    kv_batch_offset = K * Dh
    logits_score_head_offset = batch_size * K
    logits_score_batch_offset = K
    q_out_head_offset = batch_size * Dh
    q_out_batch_offset = Dh
    # print(values.flags['C_CONTIGUOUS'])
    # print(logits.flags['C_CONTIGUOUS'])
    # print(result.flags['C_CONTIGUOUS'])

    for t_idx in range(thread_num):
        start_idx = t_idx * work_per_thread
        end_idx = min(start_idx + work_per_thread, total_work)

        # Call the C++ function through the module
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

        # Split Dataset
        # attn_module.attn_output_threaded(values[start_idx:end_idx], logits[start_idx:end_idx], result[start_idx:end_idx], head_num=head_num, batch_size=batch_size, K=K, Dh=Dh,
        #                                 values_head_offset=kv_head_offset, values_batch_offset=kv_batch_offset,
        #                                 logits_head_offset=logits_score_head_offset, logits_batch_offset=logits_score_batch_offset,
        #                                 result_head_offset=q_out_head_offset, result_batch_offset=q_out_batch_offset,
        #                                 thread_id=t_idx, thread_num=thread_num, start_idx=start_idx, end_idx=end_idx)

        # attn_module.attn_output_simple(values, logits, result, head_num=head_num, batch_size=batch_size, K=K, Dh=Dh,
        #                                 values_head_offset=kv_head_offset, values_batch_offset=kv_batch_offset,
        #                                 logits_head_offset=logits_score_head_offset, logits_batch_offset=logits_score_batch_offset,
        #                                 result_head_offset=q_out_head_offset, result_batch_offset=q_out_batch_offset,
        #                                 thread_id=t_idx, thread_num=thread_num, start_idx=0, end_idx=1)

        # print(result)


if __name__ == "__main__":
    test_single_process()
