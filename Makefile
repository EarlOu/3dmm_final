CFLAGS=-c -g -fopenmp

all: main.o sift.o utils.o pgm.o
	g++ *.o -fopenmp -lOpenCL

main.o: main.cpp sift.h
	g++ $< $(CFLAGS)

sift.o: sift.cpp utils.h
	g++ $< $(CFLAGS) -I/opt/cuda/include/

utils.o: utils.cpp utils.h
	g++ $< $(CFLAGS)

pgm.o: pgm.cpp pgm.h
	g++ $< $(CFLAGS)
