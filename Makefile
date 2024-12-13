attn:
	#g++ -I/usr/local/cuda/include -o main main.cpp value_gemv.cpp test_value_gemv.cpp key_gemv.cpp test_key_gemv.cpp utils.cpp -std=c++11 -ffp-contract=off -lopenblas -Wall -pedantic -O3 -mavx -mavx2 -mfma -ffast-math -march=native -fopenmp -lnuma
	g++ -I/usr/local/cuda/include -o main main.cpp value_gemv.cpp test_value_gemv.cpp key_gemv.cpp utils.cpp -std=c++11 -ffp-contract=off -lopenblas -Wall -pedantic -O3 -mavx -mavx2 -mfma -ffast-math -march=native -fopenmp -lnuma

shift:
	g++ -o shift shift.cpp 

clean:
	rm -f main
