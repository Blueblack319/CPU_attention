BS_arr=(16 32 64 128 256)
S_arr=(2048 4096)
thread_num_arr=(8 16 32 48 64)

for S in "${S_arr[@]}"; do
    for thread_num in "${thread_num_arr[@]}"; do
        for BS in "${BS_arr[@]}"; do
            sudo ./main ${BS} ${S} 0.01 ${thread_num} 0
        done
    done
done

