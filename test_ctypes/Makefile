attn:
	g++ -I/usr/local/cuda/include -shared -fPIC -o libattn.so cpu_attention.cpp -std=c++11 -Wall -pedantic -O3 -mavx -mavx2 -mfma -ffast-math -march=native -lnuma

cppthreads:
	g++ -I/usr/local/cuda/include -shared -fPIC -o libcpp_threads.so cpp_threads.cpp -std=c++11 -Wall -pedantic -O3 -mavx -mavx2 -mfma -ffast-math -march=native -lnuma

clean:
	rm -f libattn.so

