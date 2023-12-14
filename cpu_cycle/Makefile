CC=g++
#CXXFLAGS+=-I ../tests/ -I /nfs_home/nhasabni/bin/usr/lib/habanatools/include -O2 -fopenmp -mavx512f
CXXFLAGS+=-O2 -fopenmp -mavx512f
LDFLAGS=-l gomp

$(OBJFILES:.o=.cpp)

%: %.cpp

darken_blend_cpu_clocks: darken_blend_cpu_clocks.o

.PHONY: clean

clean:
	rm *.o darken_blend_cpu_clocks
