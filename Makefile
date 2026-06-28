BUILD_DIR := build
CXX ?= g++
NVCC ?= nvcc

CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
NVCCFLAGS ?= -O3 -std=c++17 -Xcompiler -pthread

.PHONY: all clean

all: $(BUILD_DIR)/cpu_matmul $(BUILD_DIR)/gpu_matmul

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/cpu_matmul: src/cpu_matmul.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@ -pthread

$(BUILD_DIR)/gpu_matmul: src/gpu_matmul.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR) results