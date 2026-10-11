// Standalone one-op MMVQ/MMQ probe for the pinned llama.cpp runtime.
// It reads one quantized tensor from the supplied GGUF, builds exactly one
// ggml_mul_mat node, and submits that graph to the same ggml-cuda.dll used by
// llama.cpp.  CUPTI joins the NVTX range with the main kernel geometry.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <nvtx3/nvToolsExt.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const std::string & message) { if (!ok) throw std::runtime_error(message); }
static void cuda_require(cudaError_t status, const char * operation) {
    require(status == cudaSuccess, std::string(operation) + ": " + cudaGetErrorString(status));
}
static std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (const char c : value) {
        if (c == '\\') out << "\\\\";
        else if (c == '"') out << "\\\"";
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else out << c;
    }
    return out.str();
}

struct GGUFFile {
    gguf_context * meta = nullptr;
    std::string path;
    ~GGUFFile() { if (meta) gguf_free(meta); }
};

static GGUFFile open_gguf(const std::string & path) {
    gguf_init_params params{};
    params.no_alloc = true;
    GGUFFile file;
    file.meta = gguf_init_from_file(path.c_str(), params);
    file.path = path;
    require(file.meta != nullptr, "gguf_init_from_file failed");
    return file;
}

static std::vector<uint8_t> read_tensor_bytes(const GGUFFile & file, int64_t id) {
    const size_t offset = gguf_get_data_offset(file.meta) + gguf_get_tensor_offset(file.meta, id);
    const size_t size = gguf_get_tensor_size(file.meta, id);
    std::ifstream input(file.path, std::ios::binary);
    require(input.good(), "cannot open GGUF tensor data");
    input.seekg(static_cast<std::streamoff>(offset));
    std::vector<uint8_t> bytes(size);
    input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size));
    require(input.gcount() == static_cast<std::streamsize>(size), "short GGUF tensor read");
    return bytes;
}

struct TensorCase {
    const char * case_id;
    const char * format;
    ggml_type type;
    int64_t n;
    const char * preferred_name;
};
static std::vector<TensorCase> cases() {
    return {
        {"q5_0_attn_q", "q5_0", GGML_TYPE_Q5_0, 896, "blk.0.attn_q.weight"},
        {"q5_0_ffn_gate", "q5_0", GGML_TYPE_Q5_0, 4864, "blk.0.ffn_gate.weight"},
        {"q8_0_attn_v", "q8_0", GGML_TYPE_Q8_0, 128, "blk.0.attn_v.weight"},
        {"q8_0_output", "q8_0", GGML_TYPE_Q8_0, 151936, "output.weight"},
    };
}
static int64_t find_tensor(const GGUFFile & file, const TensorCase & spec) {
    const int64_t preferred = gguf_find_tensor(file.meta, spec.preferred_name);
    if (preferred >= 0 && gguf_get_tensor_type(file.meta, preferred) == spec.type) {
        const int64_t * ne = gguf_get_tensor_ne(file.meta, preferred);
        if (ne[0] == 896 && ne[1] == spec.n) return preferred;
    }
    for (int64_t id = 0, count = gguf_get_n_tensors(file.meta); id < count; ++id) {
        if (gguf_get_tensor_type(file.meta, id) != spec.type) continue;
        const int64_t * ne = gguf_get_tensor_ne(file.meta, id);
        if (ne[0] == 896 && ne[1] == spec.n) return id;
    }
    return -1;
}

struct GraphOp {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * weight = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ~GraphOp() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (backend) ggml_backend_free(backend);
        if (ctx) ggml_free(ctx);
    }
};

// The pinned backend deliberately creates synchronization-only events for its
// public device-event interface.  Wrap a timing-enabled CUDA event in the same
// private backend event object so ggml_backend_event_record() still records on
// ggml-cuda's stream, while cudaEventElapsedTime() remains available here.
static ggml_backend_event_t make_timing_event(ggml_backend_t backend) {
    auto * event = new ggml_backend_event{};
    event->device = ggml_backend_get_device(backend);
    cudaEvent_t cuda_event = nullptr;
    cuda_require(cudaEventCreateWithFlags(&cuda_event, cudaEventDefault),
                 "cudaEventCreateWithFlags(timing)");
    event->context = reinterpret_cast<void *>(cuda_event);
    return event;
}

static float timing_event_elapsed_ms(ggml_backend_event_t before,
                                     ggml_backend_event_t after) {
    float elapsed_ms = 0.0f;
    cuda_require(cudaEventElapsedTime(
                     &elapsed_ms,
                     reinterpret_cast<cudaEvent_t>(before->context),
                     reinterpret_cast<cudaEvent_t>(after->context)),
                 "cudaEventElapsedTime");
    return elapsed_ms;
}

static bool explicit_l2_flush_requested() {
    const char * value = std::getenv("HETEROLLM_MMVQ_L2_FLUSH");
    return value != nullptr && (std::string(value) == "1" || std::string(value) == "true");
}

__global__ static void mmvq_l2_flush_kernel(uint32_t * buffer, size_t words) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < words) {
        const uint32_t value = buffer[index];
        buffer[index] = value ^ 0x9e3779b9u;
    }
}

// This is an eviction attempt, not a DRAM measurement.  It allocates at least
// 4x the reported L2 capacity and performs a full device read/write pass.  All
// work is synchronized before the timing event is recorded.
static void explicit_l2_flush() {
    int device = 0;
    cuda_require(cudaGetDevice(&device), "cudaGetDevice");
    int l2_bytes = 0;
    cuda_require(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, device),
                 "cudaDeviceGetAttribute(cudaDevAttrL2CacheSize)");
    const size_t bytes = std::max<size_t>(static_cast<size_t>(l2_bytes) * 4, 4u * 1024u * 1024u);
    const size_t words = (bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    uint32_t * buffer = nullptr;
    cuda_require(cudaMalloc(reinterpret_cast<void **>(&buffer), words * sizeof(uint32_t)),
                 "cudaMalloc(L2 flush)");
    cudaError_t status = cudaMemset(buffer, 0xa5, words * sizeof(uint32_t));
    if (status == cudaSuccess) {
        const int threads = 256;
        const int blocks = static_cast<int>((words + threads - 1) / threads);
        mmvq_l2_flush_kernel<<<blocks, threads>>>(buffer, words);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) status = cudaDeviceSynchronize();
    cudaError_t free_status = cudaFree(buffer);
    cuda_require(status, "explicit L2 flush");
    cuda_require(free_status, "cudaFree(L2 flush)");
}

static GraphOp make_graph(ggml_type type, int64_t n, int m,
                          const std::vector<uint8_t> & weights,
                          const std::vector<float> & input_data) {
    GraphOp op;
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    op.ctx = ggml_init(params);
    require(op.ctx != nullptr, "ggml_init failed");
    op.weight = ggml_new_tensor_2d(op.ctx, type, 896, n);
    op.input = ggml_new_tensor_2d(op.ctx, GGML_TYPE_F32, 896, m);
    op.output = ggml_mul_mat(op.ctx, op.weight, op.input);
    require(op.weight && op.input && op.output, "failed to construct ggml_mul_mat");
    ggml_set_name(op.output, "standalone_mmvq_output");
    op.graph = ggml_new_graph(op.ctx);
    ggml_build_forward_expand(op.graph, op.output);
    op.backend = ggml_backend_cuda_init(0);
    require(op.backend != nullptr, "ggml_backend_cuda_init failed");
    op.buffer = ggml_backend_alloc_ctx_tensors_from_buft(op.ctx,
        ggml_backend_get_default_buffer_type(op.backend));
    require(op.buffer != nullptr, "CUDA tensor allocation failed");
    ggml_backend_tensor_set(op.weight, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(op.input, input_data.data(), 0, input_data.size() * sizeof(float));
    return op;
}

static std::vector<float> cpu_reference(ggml_type type, int64_t n, int m,
                                        const std::vector<uint8_t> & weights,
                                        const std::vector<float> & input_data) {
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "CPU ggml_init failed");
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, type, 896, n);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 896, m);
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr, "ggml_backend_cpu_init failed");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx,
        ggml_backend_get_default_buffer_type(backend));
    require(buffer != nullptr, "CPU tensor allocation failed");
    ggml_backend_tensor_set(weight, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "CPU reference graph failed");
    std::vector<float> result(static_cast<size_t>(n) * m);
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return result;
}

static void run_case(const GGUFFile & file, const TensorCase & spec, int m,
                     int warmups, int repeats, int & formal_counter) {
    const int64_t tid = find_tensor(file, spec);
    require(tid >= 0, std::string("exact GGUF tensor not found for ") + spec.case_id);
    const int64_t * ne = gguf_get_tensor_ne(file.meta, tid);
    require(ne[0] == 896 && ne[1] == spec.n, "unexpected GGUF shape");
    const std::vector<uint8_t> weights = read_tensor_bytes(file, tid);
    std::vector<float> input(static_cast<size_t>(896) * m);
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(0.017f * static_cast<float>(i + 1)) * 0.25f;
    GraphOp op = make_graph(spec.type, spec.n, m, weights, input);
    std::vector<float> reference;
    if (m == 1) reference = cpu_reference(spec.type, spec.n, m, weights, input);
    const std::string phase = m == 1 ? "decode" : "control";
    const std::string family = m == 1 ? "mmvq" : "mmq";
    const std::string cache_protocol = explicit_l2_flush_requested()
                                     ? "explicit_l2_eviction_attempt"
                                     : "warm_reuse_with_output_readback";
    // Reuse one graph and one input/weight allocation across warmups and
    // formal iterations.  The optional eviction pass is outside the event
    // interval and is explicitly labelled as an attempt, not a DRAM claim.
    ggml_backend_event_t before = make_timing_event(op.backend);
    ggml_backend_event_t after = make_timing_event(op.backend);
    require(before && after, "backend CUDA timing event allocation failed");
    bool validation_done = false;
    double validation_max_abs = 0.0, validation_max_rel = 0.0, validation_checksum = 0.0;
    size_t validation_count = 0;
    bool validation_finite = true, validation_numeric = false;
    for (int iteration = 0; iteration < warmups + repeats; ++iteration) {
        const bool measured = iteration >= warmups;
        const int formal_index = measured ? formal_counter++ : -1;
        ggml_backend_synchronize(op.backend);
        if (explicit_l2_flush_requested()) explicit_l2_flush();
        const std::string label = std::string("mmvq:") + spec.case_id + ":m=" + std::to_string(m) +
                                  ":formal=" + std::to_string(formal_index);
        nvtxRangePushA(label.c_str());
        double host_ms = 0.0;
        float cuda_event_ms = 0.0f;
        if (measured) {
            ggml_backend_event_record(before, op.backend);
            const auto host_start = std::chrono::steady_clock::now();
            require(ggml_backend_graph_compute(op.backend, op.graph) == GGML_STATUS_SUCCESS,
                    "CUDA graph compute failed");
            ggml_backend_event_record(after, op.backend);
            ggml_backend_event_synchronize(after);
            const auto host_stop = std::chrono::steady_clock::now();
            cuda_event_ms = timing_event_elapsed_ms(before, after);
            host_ms = std::chrono::duration<double, std::milli>(host_stop - host_start).count();
        } else {
            require(ggml_backend_graph_compute(op.backend, op.graph) == GGML_STATUS_SUCCESS,
                    "CUDA warmup graph compute failed");
            ggml_backend_synchronize(op.backend);
        }
        nvtxRangePop();
        if (!measured) continue;

        const bool validation_readback = !validation_done;
        if (validation_readback) {
            std::vector<float> result(static_cast<size_t>(spec.n) * m);
            ggml_backend_tensor_get(op.output, result.data(), 0, result.size() * sizeof(float));
            validation_count = result.size();
            for (size_t i = 0; i < result.size(); ++i) {
                validation_finite = validation_finite && std::isfinite(result[i]);
                validation_checksum += result[i];
                if (m == 1) {
                    const double abs_err = std::abs(static_cast<double>(result[i]) - reference[i]);
                    validation_max_abs = std::max(validation_max_abs, abs_err);
                    validation_max_rel = std::max(validation_max_rel, abs_err / std::max(1e-6, std::abs(static_cast<double>(reference[i]))));
                }
            }
            // CPU and CUDA use different accumulation/reordering paths for
            // these quantized dots.  A bounded absolute check is sufficient
            // for this standalone correctness gate.
            validation_numeric = validation_finite && (m != 1 || (validation_max_abs <= 1e-1));
            validation_done = true;
        }
        std::cout << "{\"schema\":\"heterollm.native-mmvq-node/v2\",\"case_id\":\"" << spec.case_id
                  << "\",\"tensor_name\":\"" << json_escape(gguf_get_tensor_name(file.meta, tid))
                  << "\",\"phase\":\"" << phase << "\",\"kernel_family\":\"" << family
                  << "\",\"weight_format\":\"" << spec.format << "\",\"shape\":[" << m << "," << spec.n << ",896]"
                  << ",\"formal_index\":" << formal_index
                  << ",\"cuda_event_recorded\":true,\"cuda_event_timing_enabled\":true"
                  << ",\"cuda_event_boundary\":\"backend_private_stream_graph\""
                  << ",\"cuda_event_elapsed_ms\":" << std::setprecision(9) << cuda_event_ms
                  << ",\"backend_event_host_elapsed_ms\":" << host_ms
                  << ",\"grid\":[0,0,0],\"block\":[0,0,0],\"dispatch_signature\":\"pending_cupti_join\""
                  << ",\"fusion_mode\":\"none\",\"cache_protocol\":\"" << cache_protocol << "\""
                  << ",\"validation_readback_performed\":" << (validation_readback ? "true" : "false")
                  << ",\"numerical_validation\":{\"passed\":" << (validation_numeric ? "true" : "false")
                  << ",\"count\":" << validation_count << ",\"max_abs\":" << validation_max_abs
                  << ",\"max_rel\":" << validation_max_rel << ",\"checksum\":" << validation_checksum << "}}\n";
        std::cout.flush();
    }
    ggml_backend_event_free(before);
    ggml_backend_event_free(after);
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: standalone_mmvq_harness model.gguf warmups repeats\n");
        return 2;
    }
    try {
        const int warmups = std::stoi(argv[2]);
        const int repeats = std::stoi(argv[3]);
        require(warmups >= 0 && repeats >= 1, "warmups must be >=0 and repeats >=1");
        std::ofstream output_file;
        std::streambuf * previous_stdout = nullptr;
        if (const char * output_path = std::getenv("HETEROLLM_MMVQ_OUTPUT")) {
            output_file.open(output_path, std::ios::out | std::ios::trunc);
            require(output_file.good(), std::string("cannot open HETEROLLM_MMVQ_OUTPUT: ") + output_path);
            previous_stdout = std::cout.rdbuf(output_file.rdbuf());
        }
        const GGUFFile file = open_gguf(argv[1]);
        int index = 0;
        const bool m1_only = [] {
            const char * value = std::getenv("HETEROLLM_MMVQ_M1_ONLY");
            return value != nullptr && (std::string(value) == "1" || std::string(value) == "true");
        }();
        const char * filter = std::getenv("HETEROLLM_MMVQ_CASE");
        for (const TensorCase & spec : cases()) {
            if (filter != nullptr && std::string(filter) != spec.case_id) continue;
            run_case(file, spec, 1, warmups, repeats, index);
            if (!m1_only) run_case(file, spec, 512, warmups, repeats, index);
        }
        if (previous_stdout != nullptr) std::cout.rdbuf(previous_stdout);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "standalone MMVQ harness failed: %s\n", error.what());
        return 1;
    }
}
