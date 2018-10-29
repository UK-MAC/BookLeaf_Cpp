# Use nvcc_wrapper from https://github.com/kokkos/nvcc_wrapper
CXX := nvcc_wrapper

# Configurable build options
DEPSDIR := $$HOME
GPUARCH := 30
USEMPI  := 1

ifeq ($(USEMPI),1)
    HOSTCXX := $(shell which mpicxx)
else
    HOSTCXX := $(shell echo $$CXX)
endif

# Set compile flags
CXXFLAGS := -ccbin $(HOSTCXX) -arch compute_$(GPUARCH) -code sm_$(GPUARCH) -O3 \
            -DNDEBUG -I./include -I$(DEPSDIR)/include --expt-extended-lambda \
            -std=c++11 -DBOOKLEAF_MESH_GENERATION

ifeq ($(USEMPI),1)
    CXXFLAGS += -DBOOKLEAF_MPI_SUPPORT -DBOOKLEAF_PARMETIS_SUPPORT \
                -DBOOKLEAF_MPI_DT_CONTEXT
endif

# Set link flags
LDFLAGS := -L$(DEPSDIR)/lib -L$(DEPSDIR)/lib64 -lyaml-cpp

ifeq ($(USEMPI),1)
    LDFLAGS += -lparmetis -lmetis -ltyphon
endif

# In/out directories
SRCDIR   := src
BUILDDIR := build-cuda

# List source files
SRC := $(shell find $(SRCDIR) -name '*.cpp')

# Remove Silo and debug source files
SRC := $(filter-out src/packages/io/driver/silo_io_driver.cpp,$(SRC))
SRC := $(filter-out src/utilities/debug/zlib_compressor.cpp,$(SRC))

# If MPI is disabled, remove the comms files
ifneq ($(USEMPI),1)
    SRC := $(filter-out src/packages/setup/partition_mesh.cpp,$(SRC))
    SRC := $(filter-out src/packages/setup/distribute_mesh.cpp,$(SRC))
    SRC := $(filter-out src/utilities/comms/partition.cpp,$(SRC))
    SRC := $(filter-out src/utilities/comms/exchange.cpp,$(SRC))
    SRC := $(filter-out src/utilities/comms/dt_reduce.cpp,$(SRC))
endif

# Determine output files
OBJ := $(SRC:.cpp=.o)
OBJ := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(OBJ))
EXE := $(BUILDDIR)/bookleaf

all: $(EXE)

$(BUILDDIR)/bookleaf: $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(BUILDDIR)/bookleaf $^

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR)
	mkdir -p $(shell dirname $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILDDIR):
	mkdir $(BUILDDIR)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)
