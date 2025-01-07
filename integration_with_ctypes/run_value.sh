#!/bin/bash
BS_arr=(64 128 256)
S_arr=(2048 4096 8192)
ratios=(0.01 0.02)

# ratio -> batch -> seq_len
for BS in "${BS_arr[@]}"; do
    for S in "${S_arr[@]}"; do
        for ratio in "${ratios[@]}"; do
            sudo cset proc --set=user --exec numactl -- --cpunodebind=0 --membind=0 python example.py --batch_size ${BS} --S_len ${S} --ratio ${ratio} --thread_num 46 --gqa
        done
    done
done
