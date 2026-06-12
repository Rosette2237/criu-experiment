#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

static std::atomic<bool> keep_running{true};

static void handle_signal(int) {
    keep_running.store(false);
}

struct Args {
    int n = 512;
    long long iters = 0;          // 0 means run until signaled
    int extra_mb = 512;           // CPU memory included in CRIU image
    int sleep_ms = 0;
    std::string csv = "results/cpu_app.csv";
};

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
        else if (k == "--csv") args.csv = next();
        else {
            std::cerr << "Unknown argument: " << k << "\n";
            std::exit(2);
        }
    }
    return args;
}

static void maybe_write_header(const std::string& path, std::ofstream& out) {
    bool need_header = true;
    if (fs::exists(path) && fs::file_size(path) > 0) {
        need_header = false;
    }
    if (need_header) {
        out << "mode,pid,iter,wall_s,iter_s,checksum,n,extra_mb\n";
    }
}

int main(int argc, char** argv) {
    std::signal(SIGTERM, handle_signal);
    std::signal(SIGINT, handle_signal);

    Args args = parse_args(argc, argv);
    fs::create_directories(fs::path(args.csv).parent_path());

    const int n = args.n;
    const size_t elems = static_cast<size_t>(n) * static_cast<size_t>(n);
    const size_t extra_floats =
        (static_cast<size_t>(args.extra_mb) * 1024ull * 1024ull) / sizeof(float);

    std::vector<float> A(elems), B(elems), C(elems, 0.0f);
    std::vector<float> extra(extra_floats, 1.0f);

    for (size_t i = 0; i < elems; ++i) {
        A[i] = static_cast<float>((i % 97) + 1) * 0.001f;
        B[i] = static_cast<float>((i % 89) + 1) * 0.002f;
    }

    std::ofstream out(args.csv, std::ios::app);
    if (!out) {
        std::cerr << "Failed to open CSV: " << args.csv << "\n";
        return 1;
    }
    maybe_write_header(args.csv, out);

    const auto start = std::chrono::steady_clock::now();
    long long iter = 0;

    std::cerr << "CPU workload pid=" << getpid()
              << " n=" << n
              << " extra_mb=" << args.extra_mb
              << " iters=" << args.iters << "\n";

    while (keep_running.load() && (args.iters == 0 || iter < args.iters)) {
        const auto iter_start = std::chrono::steady_clock::now();

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < n; ++k) {
                    sum += A[static_cast<size_t>(i) * n + k] *
                           B[static_cast<size_t>(k) * n + j];
                }
                C[static_cast<size_t>(i) * n + j] = sum;
            }
        }

        // Touch one float per 4 KiB page so CPU checkpoint size is meaningful.
        for (size_t idx = 0; idx < extra.size(); idx += 1024) {
            extra[idx] += 0.000001f * static_cast<float>(iter + 1);
        }

        const auto iter_end = std::chrono::steady_clock::now();
        const double iter_s =
            std::chrono::duration<double>(iter_end - iter_start).count();
        const double wall_s =
            std::chrono::duration<double>(iter_end - start).count();

        const float checksum =
            C[0] + C[elems / 2] + C[elems - 1] +
            (extra.empty() ? 0.0f : extra[(static_cast<size_t>(iter) * 1024) % extra.size()]);

        out << "cpu," << getpid() << "," << iter << ","
            << wall_s << "," << iter_s << "," << checksum << ","
            << n << "," << args.extra_mb << "\n";
        out.flush();

        ++iter;

        if (args.sleep_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(args.sleep_ms));
        }
    }

    std::cerr << "CPU workload exiting pid=" << getpid()
              << " final_iter=" << iter << "\n";
    return 0;
}