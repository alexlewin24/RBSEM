CC=g++
CFLAGS=-c -Wall -std=c++11 -fopenmp #-O3 #commenting out -OX optimisation because of valgrind   (de-comment it once ready for production, 3fold decrease in runtime)
LDFLAGS= -larmadillo -lpthread -llapack -lopenblas -fopenmp
# LDFLAGS= -L/usr/lib/x86_64-linux-gnu/ -fopenmp -larmadillo -lnvblas -llapack -ltrng4

SOURCES_HESS=global.cpp utils.cpp distr.cpp HESS.cpp imputation.cpp run_HESS.cpp test.cpp
OBJECTS_HESS=$(SOURCES_HESS:.cpp=.o)

all: $(SOURCES_HESS) HESS

HESS: $(OBJECTS_HESS)
	$(CC) $(OBJECTS_HESS) -o HESS_Reg $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm *.o
