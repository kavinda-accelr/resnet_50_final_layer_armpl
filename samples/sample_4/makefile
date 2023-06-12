CXX=g++
CXXFLAGS=-std=c++11 -O3 -mcpu=native -I/opt/arm/armpl_23.04_gcc-12.2/include
LDFLAGS=-L/opt/arm/armpl_23.04_gcc-12.2/lib -lastring -larmpl_mp -lamath -lm -fopenmp

all: main

main: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o $(LDFLAGS)
	$(CXX) $(CXXFLAGS) -o main.out main.o $(LDFLAGS)
	./main.out

clean:
	rm -f *.o *.out