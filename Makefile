attn:
	g++ -o main main.cpp attention_score.cpp attention_output.cpp -ffp-contract=off -lopenblas -Wall -pedantic -O3 -mavx -mavx2 -mfma -ffast-math -march=native -fopenmp

shift:
	g++ -o shift shift.cpp 

clean:
	rm -f main


