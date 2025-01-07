#!/bin/bash
BS_arr=(32 64 128 256)
K_arr=(20 40 81 163)

# ratio -> batch -> seq_len
for BS in "${BS_arr[@]}"; do
    for K in "${K_arr[@]}"; do
         sudo numactl --cpunodebind=0 --membind=0 python example.py --batch_size ${BS} --K ${K} --thread_num 46 --key_gemv
    done
done
