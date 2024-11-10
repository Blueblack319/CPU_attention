attn:
	g++ -I/usr/local/cuda/include -shared -fPIC -o libattn.so example.cpp -std=c++11 -Wall -pedantic -O3 -mavx -mavx2 -mfma -ffast-math -march=native -lnuma

clean:
	rm -f libattn.so

