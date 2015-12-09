INTERVALS = 1
DESIGN = mul
REAL = float
#REAL = double

ASMDEP = bench/$(DESIGN).c
ASMDEP += bench/$(DESIGN).range
ASM = data/$(DESIGN)/$(DESIGN).asm

#DEBUGOPT1 = -g -G
#DEBUGOPT2 = -ggdb 
#MAC = -ccbin /usr/bin/clang -Xcompiler -stdlib=libstdc++
DEBUGOPT1 = -O3 -DTHRUST_DEBUG
DEBUGOPT2 = -O3

BIN_DIR = ./bin
LIB_DIR = ./lib
LIB_ASM = $(LIB_DIR)/libasm.so
LIB_BITSLICE_CORE = $(LIB_DIR)/libbitgpu_core.so
CC_INCLUDE = -I./include -I/usr/local/cuda/include
ASM_LDFLAGS = -L$(LIB_DIR) -lasm
BITSLICE_CORE_LDFLAGS = -L$(LIB_DIR) -lbitgpu_core
LIB_LDFLAGS = $(ASM_LDFLAGS) $(BITSLICE_CORE_LDFLAGS) -lm

all: $(BIN_DIR) $(LIB_DIR) libasm libbitgpu_core range_driver bitgpu_driver prune_driver

$(BIN_DIR):
ifneq ($(BIN_DIR),)
	mkdir $(BIN_DIR)
endif

$(LIB_DIR):
ifneq ($(LIB_DIR),)
	mkdir $(LIB_DIR)
endif

libasm: 
	g++ -Wno-unused-result $(DEBUGOPT2) -DREAL=$(REAL) $(CC_INCLUDE) --shared -fPIC src/asm.cpp -o $(LIB_ASM)

libbitgpu_core: 
	g++ $(DEBUGOPT2) -DREAL=$(REAL) $(CC_INCLUDE) --shared -fPIC -std=c++0x src/bitgpu_core.cpp -o $(LIB_BITSLICE_CORE)

stuff:
	g++ $(DEBUGOPT2) -DREAL=$(REAL) $(CC_INCLUDE) src/stuff.cpp -o $(BIN_DIR)/stuff $(ASM_LDFLAGS)

range_driver:
	nvcc -arch sm_20 $(DEBUGOPT1) -DREAL=$(REAL) $(CC_INCLUDE) src/range_driver.cu src/range.cu -o $(BIN_DIR)/range_driver $(ASM_LDFLAGS)

range_mc_driver:
	nvcc -arch sm_20 $(DEBUGOPT1) -DREAL=$(REAL) $(CC_INCLUDE) src/range_driver.cu src/range.cu -o $(BIN_DIR)/range_mc_driver $(ASM_LDFLAGS) -DUSE_MC

prune_driver:
	g++  $(DEBUGOPT2) -DREAL=$(REAL) $(CC_INCLUDE) src/prune_driver.cpp src/bitgpu_core.cpp -o $(BIN_DIR)/prune_driver $(LIB_LDFLAGS)

prune_mc_driver:
	g++  $(DEBUGOPT2) -DREAL=$(REAL) $(CC_INCLUDE) -std=c++0x src/prune_mc_driver.cpp src/bitgpu_core.cpp -o $(BIN_DIR)/prune_mc_driver $(LIB_LDFLAGS) -DUSE_MC

bitgpu_driver:
	nvcc -arch sm_20 $(DEBUGOPT1) -DREAL=$(REAL) $(CC_INCLUDE) src/bitgpu_driver.cu src/bitgpu_error.cu -o $(BIN_DIR)/bitgpu_driver $(ASM_LDFLAGS)

test_enum:
	g++ $(DEBUGOPT2) -DREAL=$(REAL) $(CC_INCLUDE) src/test_enum.cpp -o $(BIN_DIR)/test_enum $(ASM_LDFLAGS)

monte_carlo_driver:
	nvcc -arch sm_20 $(DEBUGOPT1) -DREAL=$(REAL) $(CC_INCLUDE) src/monte_carlo_driver.cu src/monte_carlo_error.cu -o $(BIN_DIR)/monte_carlo_driver $(ASM_LDFLAGS) -DUSE_MC

raw_kernels_driver: 
	nvcc -arch sm_20  $(DEBUGOPT1) -DREAL=$(REAL) $(CC_INCLUDE) src/raw_kernels_driver.cu src/raw_kernels.cu -o $(BIN_DIR)/raw_kernels_driver $(ASM_LDFLAGS)

raw_kernels_openmp_driver: 
	g++ $(DEBUGOPT2) -DREAL=$(REAL) $(CC_INCLUDE) src/raw_kernels_openmp_driver.cpp src/raw_kernels.cpp -o $(BIN_DIR)/raw_kernels_openmp_driver $(ASM_LDFLAGS) -fopenmp

asm: 
	cd scripts; ./gimple_to_asm.sh ../bench/$(DESIGN).c; cd ..

gappa:
	cd scripts; ./asm_to_gappa.sh ../bench/$(DESIGN).c; gappa ../data/$(DESIGN)/$(DESIGN).g; cd ..

gappa_sub_IA:
	cd scripts; ./asm_to_gappa_sub_IA.sh ../bench/$(DESIGN).c $(INTERVALS); gappa ../data/$(DESIGN)/$(DESIGN)_sub_IA.g; cd ..

clean:
	rm -Rf ./bin/*
	rm -Rf ./lib/*
