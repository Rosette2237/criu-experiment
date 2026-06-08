# criu-experiment/Makefile

SHELL := /bin/bash

# Compiler settings
NVCC ?= nvcc
CC   ?= gcc

# Build directories
BUILD_DIR := build
BIN_DIR   := $(BUILD_DIR)/bin

# Source directories
CUDA_SRC_DIR := src/cuda_matmul
CPU_SRC_DIR  := src/cpu_baseline
COMMON_DIR   := src/common

# Output binaries
CUDA_BIN := $(BIN_DIR)/matmul_bench
CPU_BIN  := $(BIN_DIR)/cpu_sleep_loop

# CUDA architecture.
# For A100-class GPUs, sm_80 is appropriate.
# Override if needed:
#   make CUDA_ARCH=sm_80
CUDA_ARCH ?= sm_80

# Compiler flags
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) \
             -I$(COMMON_DIR) \
             -I$(CUDA_SRC_DIR)

CFLAGS ?= -O2 -std=c11 -Wall -Wextra -pedantic \
          -I$(COMMON_DIR)

.PHONY: all cuda cpu clean distclean dirs help

all: cuda cpu

dirs:
	@mkdir -p $(BIN_DIR)

cuda: dirs $(CUDA_BIN)

cpu: dirs $(CPU_BIN)

$(CUDA_BIN): $(CUDA_SRC_DIR)/matmul_bench.cu \
             $(CUDA_SRC_DIR)/matmul_config.h \
             $(COMMON_DIR)/timing.h \
             $(COMMON_DIR)/logging.h \
             $(COMMON_DIR)/signal_markers.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(CPU_BIN): $(CPU_SRC_DIR)/cpu_sleep_loop.c \
            $(COMMON_DIR)/timing.h \
            $(COMMON_DIR)/logging.h \
            $(COMMON_DIR)/signal_markers.h
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf $(BUILD_DIR)

distclean: clean
	rm -rf results/raw/*
	rm -rf results/parsed/*
	rm -rf results/logs/*
	rm -rf results/figures/*
	rm -rf checkpoint-images/*

help:
	@echo "CRIU CUDA checkpoint experiment Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make all       Build CUDA and CPU benchmark binaries"
	@echo "  make cuda      Build CUDA matrix multiplication benchmark"
	@echo "  make cpu       Build CPU-only CRIU baseline benchmark"
	@echo "  make clean     Remove build artifacts"
	@echo "  make distclean Remove build artifacts, result files, and checkpoint images"
	@echo ""
	@echo "Variables:"
	@echo "  NVCC           CUDA compiler, default: nvcc"
	@echo "  CC             C compiler, default: gcc"
	@echo "  CUDA_ARCH      CUDA architecture, default: sm_80"
	@echo ""
	@echo "Example:"
	@echo "  make all"
	@echo "  make cuda CUDA_ARCH=sm_80"