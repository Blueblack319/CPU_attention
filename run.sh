BS_arr=(16 32 64 128 256)
K_arr=(81 163 245 327 409 40 122 204 20 61 102 10 30 51)
# K_arr=(5 10 20 40)
thread_num_arr=(1 2 4 8 16 32 48 64)

for K in "${K_arr[@]}"; do
    for thread_num in "${thread_num_arr[@]}"; do
        for BS in "${BS_arr[@]}"; do
            sudo numactl --membind=0 env LD_LIBRARY_PATH=$LD_LIBRARY_PATH ./main ${BS} ${K} ${thread_num}
        done
    done
done
