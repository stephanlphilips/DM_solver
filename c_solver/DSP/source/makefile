CC = gcc
CFLAGS = -g -Wall -I ../include -I.
LIBS = -std=c++14 -lm -lstdc++
CXXFLAGS = -g -Wall -I ../include -I.
DEPS = Bessel.cpp Butterworth.cpp ChebyshevI.cpp Custom.cpp Biquad.cpp Cascade.cpp  Elliptic.cpp  Legendre.cpp  Param.cpp RBJ.cpp  State.cpp  ChebyshevII.cpp  Design.cpp  Filter.cpp PoleFilter.cpp  RootFinder.cpp

DEPS = Bessel.o Butterworth.o ChebyshevI.o Custom.o Biquad.o Cascade.o  Elliptic.o  Legendre.o  Param.o RBJ.o  State.o  ChebyshevII.o  Design.o  Filter.o PoleFilter.o  RootFinder.o

Documentation: Documentation.cpp $(DEPS)
	$(CC) $(CFLAGS)  -o Documentation Documentation.cpp $(DEPS) $(LIBS)

# gcc $(CFLAGS) 