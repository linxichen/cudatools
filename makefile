# Quick tutorial
# $@ = target
# $^ = all depdencies
# $? = dependcies more recent than target
# $< = only the first dependency

# Paths for includes I, and libraries L
ICUDA = /usr/local/cuda-7.0/include
LCUDA = /usr/local/cuda-7.0/lib64
ICUDA_MAC = /Developer/NVIDIA/CUDA-7.0/include
LCUDA_MAC = /Developer/NVIDIA/CUDA-7.0/lib
ICPP_MAC = /usr/local/include
LCPP_MAC = /usr/local/lib

SRCDIR = src
LIBDIR = lib

# Compiler for CUDA
NVCC = nvcc

# CUDA compiling options
NVCCFLAGS = -v -arch sm_30 #-use_fast_math

# Compiler for C code
CXX = g++

# Standard optimization flags to C++ compiler
CXXFLAGS = -O2 -std=c++11 -I$(ICUDA) -I$(ICUDA_MAC) -I$(ICPP_MAC) -I$(SRCDIR)

# Add CUDA libraries to C++ compiler linking process
LDLIBS += -lstdc++ -lcublas -lcurand -lcudart -larmadillo -lopenblas -llapack -L$(LCUDA) -L$(LCUDA_MAC) -L$(LCPP_MAC)

# List Executables and Objects
LIB = cudatools

all : createfolder lib$(LIB).a

createfolder :
	mkdir -p lib

# Create static library
lib$(LIB).a :devicecode.o hostcode.o
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -lib $^ -o $(LIBDIR)/$@ $(LDLIBS)
	mv *.o $(LIBDIR)

# Dlink CUDA relocatable object into executable object
# devicecode_dlink.o : devicecode.o
	# $(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(LDLIBS) -dlink $^ -o $@

# Compile CUDA code
devicecode.o : $(SRCDIR)/devicecode.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -dc $^ -o $@ $(LDLIBS)

# Compile C++ code
hostcode.o : $(SRCDIR)/hostcode.cpp
	$(CXX) -v $(CXXFLAGS) -c $^ -o $@ $(LDLIBS)

clean :
	rm -f $(LIBDIR)/*.o
	rm -f core core.*
