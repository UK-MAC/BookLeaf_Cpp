# Use nvcc_wrapper from https://github.com/kokkos/nvcc_wrapper
CXX := nvcc_wrapper

# Configurable build options
GPUARCH := 30
USEMPI  := 1

ifeq ($(USEMPI),1)
    HOSTCXX := $(shell which mpicxx)
else
    HOSTCXX := $(shell echo $$CXX)
endif

# Set compile flags
CXXFLAGS := -ccbin $(HOSTCXX) -arch compute_$(GPUARCH) -code sm_$(GPUARCH) -O3 \
            -DNDEBUG -I./include -I$$HOME/include --expt-extended-lambda \
            -DBOOKLEAF_MESH_GENERATION

ifeq ($(USEMPI),1)
    CXXFLAGS += -DBOOKLEAF_MPI_SUPPORT -DBOOKLEAF_PARMETIS_SUPPORT \
                -DBOOKLEAF_MPI_DT_CONTEXT -DBOOKLEAF_KOKKOS_CUDA_SUPPORT
endif

# Set link flags
LDFLAGS := -L$$HOME/lib -lyaml-cpp -lkokkos

ifeq ($(USEMPI),1)
    LDFLAGS += -lparmetis -lmetis -ltyphon
endif

# In/out directories
SRCDIR   := src
BUILDDIR := build-kokkos

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

# Kokkos options
KOKKOS_PATH         := external/kokkos
CUDA_PATH           := /usr/local/cuda
KOKKOS_DEVICES      := Cuda
KOKKOS_CUDA_OPTIONS := enable_lambda

all: $(EXE)

include $(KOKKOS_PATH)/Makefile.kokkos

$(BUILDDIR)/bookleaf: $(OBJ) | $(KOKKOS_LINK_DEPENDS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) \
		$(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) -o $(BUILDDIR)/bookleaf $^

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR) $(KOKKOS_CPP_DEPENDS)
	mkdir -p $(shell dirname $@)
	$(CXX) $(CXXFLAGS) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) -c -o $@ $<

$(BUILDDIR):
	mkdir $(BUILDDIR)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)
