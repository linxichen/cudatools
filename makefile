# Quick tutorial
# $@ = target
# $^ = all depdencies
# $? = dependcies more recent than target
# $< = only the first dependency

# Paths for includes I, and libraries L
ICUDA = /usr/local/cuda-7.5/include
LCUDA = /usr/local/cuda-7.5/lib64
ICUDA_MAC = /Developer/NVIDIA/CUDA-7.5/include
LCUDA_MAC = /Developer/NVIDIA/CUDA-7.5/lib
ICPP_MAC = /usr/local/include
LCPP_MAC = /usr/local/lib

SRCDIR = src
LIBDIR = lib

# Compiler for CUDA
NVCC = nvcc

# CUDA compiling options
NVCCFLAGS = -arch sm_30 #-use_fast_math

# Compiler for C code
CXX = g++

# Standard optimization flags to C++ compiler
CXXFLAGS += -O2 -std=c++11 
CXXFLAGS += -I$(ICUDA) -I$(ICUDA_MAC) -I$(ICPP_MAC) -I$(SRCDIR)

# Add CUDA libraries to C++ compiler linking process
LDLIBS += -lstdc++ -lcublas -lcurand -lcudart
LDLIBS += -lopenblas -llapack -L$(LCUDA) -L$(LCUDA_MAC) -L$(LCPP_MAC)
LDLIBS += -L$(LCUDA) -L$(LCUDA_MAC) -L$(LCPP_MAC)

# List Executables and Objects
LIB = cudatools

all : createfolder lib$(LIB).a

createfolder :
	mkdir -p lib

# Create static library
lib$(LIB).a : $(LIBDIR)/devicecode.o $(LIBDIR)/hostcode.o
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -lib $^ -o $(LIBDIR)/$@ $(LDLIBS)

# Dlink CUDA relocatable object into executable object
# devicecode_dlink.o : devicecode.o
	# $(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(LDLIBS) -dlink $^ -o $@

# Compile CUDA code
$(LIBDIR)/devicecode.o : $(SRCDIR)/devicecode.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -dc $^ -o $@ $(LDLIBS)

# Compile C++ code
$(LIBDIR)/hostcode.o : $(SRCDIR)/hostcode.cpp
	$(CXX) -v $(CXXFLAGS) -c $^ -o $@ $(LDLIBS)

clean :
	rm -f $(LIBDIR)/*.o
	rm -f core core.*
