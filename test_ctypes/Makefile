attn:
	g++ -I/usr/local/cuda/include -shared -fPIC -o libattn.so cpu_attention.cpp -std=c++11 -Wall -pedantic -O3 -mavx -mavx2 -mfma -ffast-math -march=native -lnuma

shm_test:
	g++ -I/usr/local/cuda/include -shared -fPIC -o libshared_memory.so shm_test.cpp -std=c++11 -lrt

clean:
	rm -f libattn.so
