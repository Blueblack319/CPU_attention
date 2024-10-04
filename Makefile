all:
	g++ -o main main.cpp attention_score.cpp attention_output.cpp -lopenblas -Wall -pedantic -O1 -mavx -mavx2 -mfma -ffast-math -march=native -fopenmp

clean:
	rm -f main


