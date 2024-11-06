import numpy as np
import build.attn_module as attn_module


# Initialize example data
batch_size = 32
K = 81
num_threads = 4
Dh = 128
num_head = 32

values = np.random.rand(num_head * batch_size * K * Dh).astype(np.float32, order='C')
logits = np.random.rand(num_head * batch_size * K).astype(np.float32, order='C')
result = np.zeros(num_head * batch_size * Dh, dtype=np.float32, order='C')

total_work = num_head * batch_size
work_per_thread = int(total_work / num_threads)


kv_head_offset = batch_size * K * Dh;
kv_batch_offset = K * Dh;
logits_score_head_offset = batch_size * K;
logits_score_batch_offset = K;
q_out_head_offset = batch_size * Dh;
q_out_batch_offset = Dh;
# print(values.flags['C_CONTIGUOUS'])
# print(logits.flags['C_CONTIGUOUS'])
# print(result.flags['C_CONTIGUOUS'])
  
for t_idx in range(num_threads):
    start_idx = t_idx * work_per_thread
    end_idx = min(start_idx + work_per_thread, total_work)
    # Call the C++ function through the module
    attn_module.attn_output_threaded(values, logits, result, num_head=num_head, batch_size=batch_size, K=K, Dh=Dh,
                                    values_head_offset=kv_head_offset, values_batch_offset=kv_batch_offset,
                                    logits_head_offset=logits_score_head_offset, logits_batch_offset=logits_score_batch_offset,
                                    result_head_offset=q_out_head_offset, result_batch_offset=q_out_batch_offset,
                                    thread_id=t_idx, num_threads=num_threads, start_idx=start_idx, end_idx=end_idx)
    # attn_module.attn_output_simple(values, logits, result, num_head=num_head, batch_size=batch_size, K=K, Dh=Dh,
    #                                 values_head_offset=kv_head_offset, values_batch_offset=kv_batch_offset,
    #                                 logits_head_offset=logits_score_head_offset, logits_batch_offset=logits_score_batch_offset,
    #                                 result_head_offset=q_out_head_offset, result_batch_offset=q_out_batch_offset,
    #                                 thread_id=t_idx, num_threads=num_threads, start_idx=0, end_idx=1)

    # print(result)