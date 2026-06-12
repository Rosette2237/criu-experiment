#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " code=" << static_cast<int>(err__)                    \
                      << " msg=" << cudaGetErrorString(err__) << "\n";          \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

static std::atomic<bool> keep_running{true};
static std::mutex log_mutex;

static void handle_signal(int) {
    keep_running.store(false);
}

struct Args {
    int n = 1024;
    long long iters = 0;          // per GPU worker; 0 means run until signaled
    int extra_mb = 2048;          // GPU device memory per GPU
    int sleep_ms = 0;
    std::string gpus = "0";
    std::string csv = "results/gpu_app.csv";
};

__global__ void matmul_kernel(const float* A, const float* B, float* C, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void init_extra_kernel(float* data, size_t count, float seed) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < count; i += stride) {
        data[i] = seed + static_cast<float>(i % 1024) * 0.00001f;
    }
}

__global__ void touch_pages_kernel(float* data, size_t count, long long iter) {
    size_t page = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pages = count / 1024;

    if (page < pages) {
        size_t idx = page * 1024;
        data[idx] += static_cast<float>((iter % 1000) + 1) * 0.000001f;
    }
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << k << "\n";
                std::exit(2);
            }
            return argv[++i];
        };

        if (k == "--n") args.n = std::stoi(next());
        else if (k == "--iters") args.iters = std::stoll(next());
        else if (k == "--extra-mb") args.extra_mb = std::stoi(next());
        else if (k == "--sleep-ms") args.sleep_ms = std::stoi(next());
        else if (k == "--gpus") args.gpus = next();
        else if (k == "--csv") args.csv = next();
        else {
            std::cerr << "Unknown argument: " << k << "\n";
            std::exit(2);
        }
    }
    return args;
}

static std::vector<int> parse_gpu_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(std::stoi(item));
        }
    }

    if (out.empty()) {
        std::cerr << "No GPU IDs provided\n";
        std::exit(2);
    }
    return out;
}

static void maybe_write_header(const std::string& path) {
    std::lock_guard<std::mutex> lock(log_mutex);
    bool need_header = true;
    if (fs::exists(path) && fs::file_size(path) > 0) {
        need_header = false;
    }

    if (need_header) {
        std::ofstream out(path, std::ios::app);
        out << "mode,pid,gpu,iter,wall_s,iter_s,checksum,n,extra_mb\n";
    }
}

static void worker(int gpu_id, const Args args) {
    CUDA_CHECK(cudaSetDevice(gpu_id));
    CUDA_CHECK(cudaFree(nullptr));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu_id));

    const int n = args.n;
    const size_t elems = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t bytes = elems * sizeof(float);
    const size_t extra_floats =
        (static_cast<size_t>(args.extra_mb) * 1024ull * 1024ull) / sizeof(float);

    std::vector<float> hA(elems), hB(elems);
    for (size_t i = 0; i < elems; ++i) {
        hA[i] = static_cast<float>(((i + gpu_id) % 97) + 1) * 0.001f;
        hB[i] = static_cast<float>(((i + 3 * gpu_id) % 89) + 1) * 0.002f;
    }

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;
    float* dExtra = nullptr;

    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMalloc(&dExtra, extra_floats * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    int init_blocks = 4096;
    int threads = 256;
    init_extra_kernel<<<init_blocks, threads>>>(dExtra, extra_floats, static_cast<float>(gpu_id + 1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        std::lock_guard<std::mutex> lock(log_mutex);
        std::cerr << "GPU worker pid=" << getpid()
                  << " gpu=" << gpu_id
                  << " name=" << prop.name
                  << " n=" << n
                  << " extra_mb=" << args.extra_mb
                  << " iters=" << args.iters << "\n";
    }

    const dim3 block(16, 16);
    const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    const size_t pages = extra_floats / 1024;
    const int page_threads = 256;
    const int page_blocks = static_cast<int>((pages + page_threads - 1) / page_threads);

    const auto start = std::chrono::steady_clock::now();
    long long iter = 0;

    while (keep_running.load() && (args.iters == 0 || iter < args.iters)) {
        const auto iter_start = std::chrono::steady_clock::now();

        matmul_kernel<<<grid, block>>>(dA, dB, dC, n);
        touch_pages_kernel<<<page_blocks, page_threads>>>(dExtra, extra_floats, iter);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float sample[4] = {0, 0, 0, 0};
        CUDA_CHECK(cudaMemcpy(&sample[0], dC, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&sample[1], dC + elems / 2, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&sample[2], dC + elems - 1, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&sample[3], dExtra + ((static_cast<size_t>(iter) * 1024) % extra_floats),
                              sizeof(float), cudaMemcpyDeviceToHost));

        const auto iter_end = std::chrono::steady_clock::now();
        const double iter_s =
            std::chrono::duration<double>(iter_end - iter_start).count();
        const double wall_s =
            std::chrono::duration<double>(iter_end - start).count();

        const float checksum = sample[0] + sample[1] + sample[2] + sample[3];

        {
            std::lock_guard<std::mutex> lock(log_mutex);
            std::ofstream out(args.csv, std::ios::app);
            out << "gpu," << getpid() << "," << gpu_id << "," << iter << ","
                << wall_s << "," << iter_s << "," << checksum << ","
                << n << "," << args.extra_mb << "\n";
            out.flush();
        }

        ++iter;

        if (args.sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(args.sleep_ms));
        }
    }

    CUDA_CHECK(cudaFree(dExtra));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));
}

int main(int argc, char** argv) {
    std::signal(SIGTERM, handle_signal);
    std::signal(SIGINT, handle_signal);

    Args args = parse_args(argc, argv);
    fs::create_directories(fs::path(args.csv).parent_path());
    maybe_write_header(args.csv);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    std::vector<int> gpus = parse_gpu_list(args.gpus);
    for (int gpu : gpus) {
        if (gpu < 0 || gpu >= device_count) {
            std::cerr << "Invalid GPU ID " << gpu
                      << "; detected device_count=" << device_count << "\n";
            return 2;
        }
    }

    std::cerr << "GPU workload pid=" << getpid()
              << " gpu_list=" << args.gpus
              << " detected_gpus=" << device_count << "\n";

    std::vector<std::thread> threads;
    for (int gpu : gpus) {
        threads.emplace_back(worker, gpu, args);
    }

    for (auto& t : threads) {
        t.join();
    }

    std::cerr << "GPU workload exiting pid=" << getpid() << "\n";
    return 0;
}