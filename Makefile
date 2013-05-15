CFLAGS=-c -g

all: main.o sift.o utils.o pgm.o
	g++ *.o

main.o: main.cpp sift.h
	g++ $< $(CFLAGS)

sift.o: sift.cpp utils.h
	g++ $< $(CFLAGS)

utils.o: utils.cpp utils.h
	g++ $< $(CFLAGS)

pgm.o: pgm.cpp pgm.h
	g++ $< $(CFLAGS)
