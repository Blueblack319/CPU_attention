# Unit Test 

Sparse CPU Attention currently supports only Value GEMV. Key GEMV is not supported. \
Key GEMV implementation needs to be modified.

## Commands

```sh
sudo apt-get update
sudo apt-get install libblas-dev libopenblas-dev numactl libnuma-dev
# batch_size seq_len topk-ratio thread_num [Value GEMV / Key GEMV]
sudo ./main 256 4096 0.01 46 0
```