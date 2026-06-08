# Experiment Design

This document describes the benchmark design for testing CRIU checkpoint/restore behavior with a CUDA matrix multiplication workload.

## Goal

The goal is to evaluate whether a CUDA process can be checkpointed and restored under CRIU with NVIDIA GPU checkpoint support, and to measure the overhead of checkpointing and restoring a CUDA workload.

The experiment should answer:

1. Does the CUDA process checkpoint successfully?
2. Does the CUDA process restore successfully?
3. Does the restored CUDA process continue execution correctly?
4. How long does checkpointing take?
5. How long does restoring take?
6. How large are the CRIU checkpoint images?
7. How does overhead scale with matrix size and GPU memory allocation?

## Main workload

The main workload is a simple CUDA matrix multiplication benchmark.

The program performs repeated matrix multiplication:

```text
C = A x B