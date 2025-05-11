# OpenCLearning

A GPU benchmarking application written in Rust that demonstrates OpenCL parallel computing concepts.

## Purpose

This project benchmarks GPU performance with different work group configurations by running a compute-intensive OpenCL kernel and measuring its execution time. The results are plotted as a chart to visualize how different thread configurations affect performance.

## Features

- **GPU Benchmarking**: Test GPU performance with various global and local work sizes
- **Visualization**: Generate plots comparing performance across different configurations
- **OpenCL Concepts**: Demonstrates key OpenCL programming concepts including:
  - Work groups and work items
  - Global and local thread indexing
  - GPU memory management

## GPU Architecture Concepts

- **SM/CU**: NVIDIA uses Streaming Multiprocessors (SM), AMD uses Compute Units (CU)
- **Warps/Wavefronts**: Groups of threads that execute in lockstep (typically 32 threads on NVIDIA)
- **SIMD**: Single Instruction Multiple Data - same operation on multiple data elements simultaneously
- **SIMT**: Single Instruction Multiple Thread - NVIDIA's extension of SIMD that allows threads some independence

## Installation

### Prerequisites

- Rust and Cargo
- OpenCL development libraries

```bash
# Debian/Ubuntu
sudo apt-get install ocl-icd-opencl-dev opencl-headers

# For NVIDIA GPUs
sudo apt-get install nvidia-opencl-dev

# For AMD GPUs
sudo apt-get install mesa-opencl-icd