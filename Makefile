CC=g++
OTHERDIR=rBSEM/src
VPATH=$(OTHERDIR)
CFLAGS=-c -Wall -std=c++11 -fopenmp -I$(OTHERDIR)/ #-O3 #commenting out -OX optimisation because of valgrind   (de-comment it once ready for production, 3fold decrease in runtime)
LDFLAGS=-larmadillo -lpthread -lopenblas -fopenmp -lboost_system -lboost_filesystem #-llapack

# LDFLAGS= -L/usr/lib/x86_64-linux-gnu/ -fopenmp -larmadillo -lnvblas -llapack -ltrng4

SOURCES_HESS=$(OTHERDIR)/global.cpp $(OTHERDIR)/utils.cpp $(OTHERDIR)/distr.cpp $(OTHERDIR)/imputation.cpp $(OTHERDIR)/HESS.cpp $(OTHERDIR)/run_HESS.cpp test.cpp
OBJECTS_HESS=$(SOURCES_HESS:.cpp=.o)

all: $(SOURCES_HESS) HESS

HESS: $(OBJECTS_HESS)
	$(CC) $(OBJECTS_HESS) -o HESS_Reg $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm $(OTHERDIR)/*.o
	rm *.o
	rm HESS_Reg
