// host_u250_bert_fake.cpp
// Host OpenCL/XRT cho thiết kế split-kernel BERT-Base trên Alveo U250.
// - Tu dong tim card U250 va nap xclbin.
// - Khoi tao du lieu gia on dinh so.
// - Chay SEQ_LEN = 128, attention mask mo du 128 token.
// - Chay du 12 encoder layer theo pipeline 6 kernel noi AXI-Stream.
// - Dung out-of-order queue de tranh deadlock stream.
// - Do thoi gian nap xclbin, H2D, tung layer, tong inference va D2H.
//
// Build vi du (XRT/Vitis 2021.2+):
//   g++ -std=c++14 -O2 -Wall host_u250_bert_fake.cpp -o host_u250_bert_fake
//       -I$XILINX_XRT/include -L$XILINX_XRT/lib -lOpenCL -pthread
//
// Run:
//   source /opt/xilinx/xrt/setup.sh
//   ./host_u250_bert_fake path/to/bert.xclbin [so_lan_chay]

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl2.hpp>
#include <CL/cl_ext_xilinx.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace cfg {
constexpr int NUM_LAYERS = 12;
constexpr int SEQ_LEN = 128;
constexpr int HIDDEN = 768;
constexpr int FFN_DIM = 3072;
constexpr int VOCAB_SIZE = 30522;
constexpr int PACK_SIZE = 16;

constexpr std::size_t TOKEN_FLOATS =
    static_cast<std::size_t>(SEQ_LEN) * HIDDEN;             // 98,304
constexpr std::size_t TOKEN_EMB_FLOATS =
    static_cast<std::size_t>(VOCAB_SIZE) * HIDDEN;          // 23,440,896
constexpr std::size_t POS_EMB_FLOATS = TOKEN_FLOATS;
constexpr std::size_t SEG_EMB_FLOATS = 2ULL * HIDDEN;

constexpr std::size_t ATTN_W_FLOATS_PER_LAYER =
    static_cast<std::size_t>(HIDDEN) * HIDDEN;              // 589,824
constexpr std::size_t ATTN_B_FLOATS_PER_LAYER = HIDDEN;
constexpr std::size_t ATTN_W_FLOATS_ALL =
    static_cast<std::size_t>(NUM_LAYERS) * ATTN_W_FLOATS_PER_LAYER;
constexpr std::size_t ATTN_B_FLOATS_ALL =
    static_cast<std::size_t>(NUM_LAYERS) * ATTN_B_FLOATS_PER_LAYER;

constexpr std::size_t FFN_UP_W_FLOATS_PER_LAYER =
    static_cast<std::size_t>(HIDDEN) * FFN_DIM;             // 2,359,296
constexpr std::size_t FFN_UP_B_FLOATS_PER_LAYER = FFN_DIM;
constexpr std::size_t FFN_DN_W_FLOATS_PER_LAYER =
    static_cast<std::size_t>(FFN_DIM) * HIDDEN;             // 2,359,296
constexpr std::size_t FFN_DN_B_FLOATS_PER_LAYER = HIDDEN;
constexpr std::size_t FFN_UP_W_FLOATS_ALL =
    static_cast<std::size_t>(NUM_LAYERS) * FFN_UP_W_FLOATS_PER_LAYER;
constexpr std::size_t FFN_UP_B_FLOATS_ALL =
    static_cast<std::size_t>(NUM_LAYERS) * FFN_UP_B_FLOATS_PER_LAYER;
constexpr std::size_t FFN_DN_W_FLOATS_ALL =
    static_cast<std::size_t>(NUM_LAYERS) * FFN_DN_W_FLOATS_PER_LAYER;
constexpr std::size_t FFN_DN_B_FLOATS_ALL =
    static_cast<std::size_t>(NUM_LAYERS) * FFN_DN_B_FLOATS_PER_LAYER;

constexpr std::size_t NORM_FLOATS_ALL =
    static_cast<std::size_t>(NUM_LAYERS) * HIDDEN;

static_assert(SEQ_LEN == 128, "Host nay duoc cau hinh cho sequence length 128");
static_assert((HIDDEN % PACK_SIZE) == 0, "HIDDEN phai chia het cho 16");
} // namespace cfg

// CL_MEM_USE_HOST_PTR tren Xilinx/AMD nen dung bo nho host can hang trang 4 KiB.
template <typename T>
class AlignedBuffer {
public:
    explicit AlignedBuffer(std::size_t count, bool zero_initialize = true)
        : count_(count), ptr_(nullptr) {
        if (count_ == 0) {
            return;
        }
        const std::size_t bytes = count_ * sizeof(T);
        void* raw = nullptr;
        const int rc = posix_memalign(&raw, 4096, bytes);
        if (rc != 0 || raw == nullptr) {
            throw std::bad_alloc();
        }
        ptr_ = static_cast<T*>(raw);
        if (zero_initialize) {
            std::memset(ptr_, 0, bytes);
        }
    }

    ~AlignedBuffer() { std::free(ptr_); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

    T& operator[](std::size_t i) { return ptr_[i]; }
    const T& operator[](std::size_t i) const { return ptr_[i]; }

    void fill(const T& value) { std::fill(ptr_, ptr_ + count_, value); }

private:
    std::size_t count_;
    T* ptr_;
};

struct HostData {
    AlignedBuffer<std::int32_t> input_ids{cfg::SEQ_LEN};
    AlignedBuffer<std::int32_t> token_type_ids{cfg::SEQ_LEN};
    AlignedBuffer<std::int32_t> attention_mask{cfg::SEQ_LEN};

    AlignedBuffer<float> token_emb{cfg::TOKEN_EMB_FLOATS};
    AlignedBuffer<float> pos_emb{cfg::POS_EMB_FLOATS};
    AlignedBuffer<float> seg_emb{cfg::SEG_EMB_FLOATS};
    AlignedBuffer<float> emb_gamma{cfg::HIDDEN};
    AlignedBuffer<float> emb_beta{cfg::HIDDEN};

    AlignedBuffer<float> hidden_ping{cfg::TOKEN_FLOATS};
    AlignedBuffer<float> hidden_pong{cfg::TOKEN_FLOATS};
    AlignedBuffer<std::uint32_t> hidden_ready{1};

    AlignedBuffer<float> attn_q_w{cfg::ATTN_W_FLOATS_ALL};
    AlignedBuffer<float> attn_q_b{cfg::ATTN_B_FLOATS_ALL};
    AlignedBuffer<float> attn_k_w{cfg::ATTN_W_FLOATS_ALL};
    AlignedBuffer<float> attn_k_b{cfg::ATTN_B_FLOATS_ALL};
    AlignedBuffer<float> attn_v_w{cfg::ATTN_W_FLOATS_ALL};
    AlignedBuffer<float> attn_v_b{cfg::ATTN_B_FLOATS_ALL};
    AlignedBuffer<float> attn_o_w{cfg::ATTN_W_FLOATS_ALL};
    AlignedBuffer<float> attn_o_b{cfg::ATTN_B_FLOATS_ALL};
    AlignedBuffer<float> attn_norm_gamma{cfg::NORM_FLOATS_ALL};
    AlignedBuffer<float> attn_norm_beta{cfg::NORM_FLOATS_ALL};

    AlignedBuffer<float> ffn_up_w{cfg::FFN_UP_W_FLOATS_ALL};
    AlignedBuffer<float> ffn_up_b{cfg::FFN_UP_B_FLOATS_ALL};
    AlignedBuffer<float> ffn_down_w{cfg::FFN_DN_W_FLOATS_ALL};
    AlignedBuffer<float> ffn_down_b{cfg::FFN_DN_B_FLOATS_ALL};
    AlignedBuffer<float> ffn_norm_gamma{cfg::NORM_FLOATS_ALL};
    AlignedBuffer<float> ffn_norm_beta{cfg::NORM_FLOATS_ALL};
};

struct DeviceBuffers {
    cl::Buffer input_ids;
    cl::Buffer token_type_ids;
    cl::Buffer attention_mask;

    cl::Buffer token_emb;
    cl::Buffer pos_emb;
    cl::Buffer seg_emb;
    cl::Buffer emb_gamma;
    cl::Buffer emb_beta;

    cl::Buffer hidden_ping;
    cl::Buffer hidden_pong;
    cl::Buffer hidden_ready;

    cl::Buffer attn_q_w;
    cl::Buffer attn_q_b;
    cl::Buffer attn_k_w;
    cl::Buffer attn_k_b;
    cl::Buffer attn_v_w;
    cl::Buffer attn_v_b;
    cl::Buffer attn_o_w;
    cl::Buffer attn_o_b;
    cl::Buffer attn_norm_gamma;
    cl::Buffer attn_norm_beta;

    cl::Buffer ffn_up_w;
    cl::Buffer ffn_up_b;
    cl::Buffer ffn_down_w;
    cl::Buffer ffn_down_b;
    cl::Buffer ffn_norm_gamma;
    cl::Buffer ffn_norm_beta;
};

struct Kernels {
    cl::Kernel embedding_prep;
    cl::Kernel qkv;
    cl::Kernel attn_core;
    cl::Kernel attn_out_norm;
    cl::Kernel ffn_up_gelu;
    cl::Kernel ffn_down_norm;
};

struct Runtime {
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
};

static std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

static std::vector<unsigned char> read_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Khong mo duoc xclbin: " + path);
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("File xclbin rong: " + path);
    }
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> data(static_cast<std::size_t>(size));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Khong doc duoc day du xclbin: " + path);
    }
    return data;
}

static std::vector<cl::Device> find_accelerator_devices() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    std::vector<cl::Device> result;
    for (const auto& platform : platforms) {
        const std::string vendor = lower_copy(platform.getInfo<CL_PLATFORM_VENDOR>());
        const std::string name = lower_copy(platform.getInfo<CL_PLATFORM_NAME>());
        if (vendor.find("xilinx") == std::string::npos &&
            vendor.find("amd") == std::string::npos &&
            name.find("xilinx") == std::string::npos &&
            name.find("amd") == std::string::npos) {
            continue;
        }

        std::vector<cl::Device> devices;
        try {
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
        } catch (const cl::Error&) {
            continue;
        }
        result.insert(result.end(), devices.begin(), devices.end());
    }

    // U250 duoc uu tien truoc; cac accelerator khac duoc giu lai de bao loi xclbin ro rang.
    std::stable_sort(result.begin(), result.end(), [](const cl::Device& a, const cl::Device& b) {
        const bool a_u250 = lower_copy(a.getInfo<CL_DEVICE_NAME>()).find("u250") != std::string::npos;
        const bool b_u250 = lower_copy(b.getInfo<CL_DEVICE_NAME>()).find("u250") != std::string::npos;
        return a_u250 && !b_u250;
    });
    return result;
}

static Runtime program_first_compatible_device(
    const std::vector<unsigned char>& xclbin,
    double& program_ms) {

    const auto devices = find_accelerator_devices();
    if (devices.empty()) {
        throw std::runtime_error(
            "Khong tim thay accelerator Xilinx/AMD. Hay kiem tra XRT, driver va lenh xbutil examine.");
    }

    std::cout << "Tim thay " << devices.size() << " accelerator:\n";
    for (std::size_t i = 0; i < devices.size(); ++i) {
        std::cout << "  [" << i << "] " << devices[i].getInfo<CL_DEVICE_NAME>() << '\n';
    }

    cl::Program::Binaries binaries{{xclbin.data(), xclbin.size()}};
    std::string last_error;

    for (const auto& device : devices) {
        const std::string device_name = device.getInfo<CL_DEVICE_NAME>();
        std::cout << "Thu nap xclbin vao: " << device_name << " ...\n";
        try {
            const auto t0 = std::chrono::steady_clock::now();
            cl::Context context(device);
            cl::Program program(context, std::vector<cl::Device>{device}, binaries);
            cl::CommandQueue queue(
                context,
                device,
                CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
            const auto t1 = std::chrono::steady_clock::now();

            program_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            return Runtime{device, context, queue, program};
        } catch (const cl::Error& e) {
            last_error = std::string(e.what()) + " (OpenCL error " + std::to_string(e.err()) + ")";
            std::cerr << "  Khong tuong thich: " << last_error << '\n';
        }
    }

    throw std::runtime_error("Khong card nao nap duoc xclbin. Loi cuoi: " + last_error);
}

template <typename T>
static cl::Buffer make_bank_buffer(
    const cl::Context& context,
    AlignedBuffer<T>& host,
    cl_mem_flags access_flags,
    unsigned bank) {

    cl_mem_ext_ptr_t ext{};
    ext.obj = host.data();
    ext.param = nullptr;
    ext.flags = static_cast<unsigned long>(bank) | XCL_MEM_TOPOLOGY;

    return cl::Buffer(
        context,
        access_flags | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX,
        host.bytes(),
        &ext);
}

static void initialize_fake_data(HostData& h) {
    // Chay du 128 token, khong padding: attention_mask tat ca bang 1.
    for (int s = 0; s < cfg::SEQ_LEN; ++s) {
        h.input_ids[s] = 1000 + s;                  // hop le trong vocab 30,522
        h.token_type_ids[s] = (s < 64) ? 0 : 1;    // 2 segment, moi segment 64 token
        h.attention_mask[s] = 1;
    }

    // Chi dien cac dong token embedding thuc su duoc truy cap.
    for (int s = 0; s < cfg::SEQ_LEN; ++s) {
        const int token_id = h.input_ids[s];
        for (int d = 0; d < cfg::HIDDEN; ++d) {
            const int centered = ((s * 13 + d * 7) % 101) - 50;
            h.token_emb[static_cast<std::size_t>(token_id) * cfg::HIDDEN + d] =
                static_cast<float>(centered) * 0.001f;
        }
    }

    for (int s = 0; s < cfg::SEQ_LEN; ++s) {
        for (int d = 0; d < cfg::HIDDEN; ++d) {
            const int centered = ((s * 5 + d * 3) % 67) - 33;
            h.pos_emb[static_cast<std::size_t>(s) * cfg::HIDDEN + d] =
                static_cast<float>(centered) * 0.0005f;
        }
    }

    for (int seg = 0; seg < 2; ++seg) {
        for (int d = 0; d < cfg::HIDDEN; ++d) {
            const int centered = ((seg + 1) * (d % 17)) - 8;
            h.seg_emb[static_cast<std::size_t>(seg) * cfg::HIDDEN + d] =
                static_cast<float>(centered) * 0.0002f;
        }
    }

    // Moi buffer lon da duoc zero trong constructor:
    // weight = 0, bias = 0, beta = 0.
    // Gamma = 1 giup LayerNorm hoat dong dung va tranh output bi ep ve 0.
    h.emb_gamma.fill(1.0f);
    h.attn_norm_gamma.fill(1.0f);
    h.ffn_norm_gamma.fill(1.0f);
    h.hidden_ready[0] = 0;
}

static DeviceBuffers create_device_buffers(const cl::Context& context, HostData& h) {
    // Bank mapping phai khop system_optimized.cfg da link vao xclbin.
    DeviceBuffers d;

    // DDR[0]: input/mask, hidden ping-pong, token dong bo, Q weights.
    d.input_ids = make_bank_buffer(context, h.input_ids, CL_MEM_READ_ONLY, 0);
    d.token_type_ids = make_bank_buffer(context, h.token_type_ids, CL_MEM_READ_ONLY, 0);
    d.attention_mask = make_bank_buffer(context, h.attention_mask, CL_MEM_READ_ONLY, 0);
    d.hidden_ping = make_bank_buffer(context, h.hidden_ping, CL_MEM_READ_WRITE, 0);
    d.hidden_pong = make_bank_buffer(context, h.hidden_pong, CL_MEM_READ_WRITE, 0);
    d.hidden_ready = make_bank_buffer(context, h.hidden_ready, CL_MEM_READ_WRITE, 0);
    d.attn_q_w = make_bank_buffer(context, h.attn_q_w, CL_MEM_READ_ONLY, 0);
    d.attn_q_b = make_bank_buffer(context, h.attn_q_b, CL_MEM_READ_ONLY, 0);

    // DDR[1]: embedding va K weights.
    d.token_emb = make_bank_buffer(context, h.token_emb, CL_MEM_READ_ONLY, 1);
    d.pos_emb = make_bank_buffer(context, h.pos_emb, CL_MEM_READ_ONLY, 1);
    d.seg_emb = make_bank_buffer(context, h.seg_emb, CL_MEM_READ_ONLY, 1);
    d.emb_gamma = make_bank_buffer(context, h.emb_gamma, CL_MEM_READ_ONLY, 1);
    d.emb_beta = make_bank_buffer(context, h.emb_beta, CL_MEM_READ_ONLY, 1);
    d.attn_k_w = make_bank_buffer(context, h.attn_k_w, CL_MEM_READ_ONLY, 1);
    d.attn_k_b = make_bank_buffer(context, h.attn_k_b, CL_MEM_READ_ONLY, 1);

    // DDR[2]: V weights va FFN-up.
    d.attn_v_w = make_bank_buffer(context, h.attn_v_w, CL_MEM_READ_ONLY, 2);
    d.attn_v_b = make_bank_buffer(context, h.attn_v_b, CL_MEM_READ_ONLY, 2);
    d.ffn_up_w = make_bank_buffer(context, h.ffn_up_w, CL_MEM_READ_ONLY, 2);
    d.ffn_up_b = make_bank_buffer(context, h.ffn_up_b, CL_MEM_READ_ONLY, 2);

    // DDR[3]: attention output, FFN-down va cac tham so LayerNorm.
    d.attn_o_w = make_bank_buffer(context, h.attn_o_w, CL_MEM_READ_ONLY, 3);
    d.attn_o_b = make_bank_buffer(context, h.attn_o_b, CL_MEM_READ_ONLY, 3);
    d.attn_norm_gamma = make_bank_buffer(context, h.attn_norm_gamma, CL_MEM_READ_ONLY, 3);
    d.attn_norm_beta = make_bank_buffer(context, h.attn_norm_beta, CL_MEM_READ_ONLY, 3);
    d.ffn_down_w = make_bank_buffer(context, h.ffn_down_w, CL_MEM_READ_ONLY, 3);
    d.ffn_down_b = make_bank_buffer(context, h.ffn_down_b, CL_MEM_READ_ONLY, 3);
    d.ffn_norm_gamma = make_bank_buffer(context, h.ffn_norm_gamma, CL_MEM_READ_ONLY, 3);
    d.ffn_norm_beta = make_bank_buffer(context, h.ffn_norm_beta, CL_MEM_READ_ONLY, 3);

    return d;
}

static Kernels create_kernels(const cl::Program& program) {
    return Kernels{
        cl::Kernel(program, "bert_embedding_prep_kernel"),
        cl::Kernel(program, "bert_qkv_kernel"),
        cl::Kernel(program, "bert_attn_core_kernel"),
        cl::Kernel(program, "bert_attn_out_norm_kernel"),
        cl::Kernel(program, "bert_ffn_up_gelu_kernel"),
        cl::Kernel(program, "bert_ffn_down_norm_kernel")};
}

static void set_layer_args(Kernels& k, DeviceBuffers& d, int layer) {
    const bool first_layer = (layer == 0);
    const bool even_layer = ((layer & 1) == 0);

    // Layer 0: embedding -> ping, FFN-down -> pong.
    // Layer 1: hidden_in pong, hidden_out ping, ... ping-pong tiep tuc.
    cl::Buffer& hidden_in = even_layer ? d.hidden_ping : d.hidden_pong;
    cl::Buffer& hidden_out = even_layer ? d.hidden_pong : d.hidden_ping;
    cl::Buffer& qkv_input = first_layer ? d.hidden_ping : hidden_in;
    cl::Buffer& attention_residual = first_layer ? d.hidden_ping : hidden_in;

    // AXI-Stream ports da duoc noi trong xclbin nen khong setArg tu host.
    // Kernel 0: bert_embedding_prep_kernel
    k.embedding_prep.setArg(0, d.input_ids);
    k.embedding_prep.setArg(1, d.token_type_ids);
    k.embedding_prep.setArg(2, d.token_emb);
    k.embedding_prep.setArg(3, d.pos_emb);
    k.embedding_prep.setArg(4, d.seg_emb);
    k.embedding_prep.setArg(5, d.emb_gamma);
    k.embedding_prep.setArg(6, d.emb_beta);
    k.embedding_prep.setArg(7, d.hidden_ping);
    k.embedding_prep.setArg(8, d.hidden_ready);
    k.embedding_prep.setArg(9, layer);

    // Kernel 1: bert_qkv_kernel
    k.qkv.setArg(0, qkv_input);
    k.qkv.setArg(1, d.hidden_ready);
    k.qkv.setArg(2, d.attn_q_w);
    k.qkv.setArg(3, d.attn_q_b);
    k.qkv.setArg(4, d.attn_k_w);
    k.qkv.setArg(5, d.attn_k_b);
    k.qkv.setArg(6, d.attn_v_w);
    k.qkv.setArg(7, d.attn_v_b);
    k.qkv.setArg(8, layer);

    // Kernel 2: bert_attn_core_kernel
    k.attn_core.setArg(0, d.attention_mask);

    // Kernel 3: bert_attn_out_norm_kernel
    k.attn_out_norm.setArg(0, attention_residual);
    k.attn_out_norm.setArg(1, d.hidden_ready);
    k.attn_out_norm.setArg(2, d.attn_o_w);
    k.attn_out_norm.setArg(3, d.attn_o_b);
    k.attn_out_norm.setArg(4, d.attn_norm_gamma);
    k.attn_out_norm.setArg(5, d.attn_norm_beta);
    k.attn_out_norm.setArg(6, layer);

    // Kernel 4: bert_ffn_up_gelu_kernel
    k.ffn_up_gelu.setArg(0, d.ffn_up_w);
    k.ffn_up_gelu.setArg(1, d.ffn_up_b);
    k.ffn_up_gelu.setArg(2, layer);

    // Kernel 5: bert_ffn_down_norm_kernel
    k.ffn_down_norm.setArg(0, d.ffn_down_w);
    k.ffn_down_norm.setArg(1, d.ffn_down_b);
    k.ffn_down_norm.setArg(2, d.ffn_norm_gamma);
    k.ffn_down_norm.setArg(3, d.ffn_norm_beta);
    k.ffn_down_norm.setArg(4, hidden_out);
    k.ffn_down_norm.setArg(5, layer);
}

static double event_duration_ms(const cl::Event& event) {
    const cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    const cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
    return static_cast<double>(end - start) * 1.0e-6;
}

struct LayerTiming {
    double wall_ms = 0.0;
    double embedding_ms = 0.0;
    double qkv_ms = 0.0;
    double attn_core_ms = 0.0;
    double attn_out_ms = 0.0;
    double ffn_up_ms = 0.0;
    double ffn_down_ms = 0.0;
};

static LayerTiming run_one_layer(
    cl::CommandQueue& queue,
    Kernels& k,
    DeviceBuffers& d,
    int layer) {

    set_layer_args(k, d, layer);

    cl::Event e_embedding;
    cl::Event e_qkv;
    cl::Event e_attn;
    cl::Event e_attn_out;
    cl::Event e_ffn_up;
    cl::Event e_ffn_down;

    const auto wall_start = std::chrono::steady_clock::now();

    // Launch downstream -> upstream. Moi CU se block tren stream/token cua no,
    // nhung tat ca CU da duoc kich hoat truoc khi producer bat dau day FIFO.
    // Khong them event dependency giua cac kernel stream: lam vay co the deadlock.
    queue.enqueueTask(k.ffn_down_norm, nullptr, &e_ffn_down);
    queue.enqueueTask(k.ffn_up_gelu, nullptr, &e_ffn_up);
    queue.enqueueTask(k.attn_out_norm, nullptr, &e_attn_out);
    queue.enqueueTask(k.attn_core, nullptr, &e_attn);
    queue.enqueueTask(k.qkv, nullptr, &e_qkv);
    queue.enqueueTask(k.embedding_prep, nullptr, &e_embedding);
    queue.flush();

    // FFN-down la consumer cuoi pipeline. Khi no xong, hidden_out cua layer da san sang.
    e_ffn_down.wait();
    e_ffn_up.wait();
    e_attn_out.wait();
    e_attn.wait();
    e_qkv.wait();
    e_embedding.wait();

    const auto wall_end = std::chrono::steady_clock::now();

    LayerTiming t;
    t.wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    t.embedding_ms = event_duration_ms(e_embedding);
    t.qkv_ms = event_duration_ms(e_qkv);
    t.attn_core_ms = event_duration_ms(e_attn);
    t.attn_out_ms = event_duration_ms(e_attn_out);
    t.ffn_up_ms = event_duration_ms(e_ffn_up);
    t.ffn_down_ms = event_duration_ms(e_ffn_down);
    return t;
}

static std::vector<cl::Memory> build_h2d_list(DeviceBuffers& d) {
    // hidden_ping/pong la output trung gian nen khong can migrate luc dau.
    return std::vector<cl::Memory>{
        d.input_ids,
        d.token_type_ids,
        d.attention_mask,
        d.hidden_ready,
        d.token_emb,
        d.pos_emb,
        d.seg_emb,
        d.emb_gamma,
        d.emb_beta,
        d.attn_q_w,
        d.attn_q_b,
        d.attn_k_w,
        d.attn_k_b,
        d.attn_v_w,
        d.attn_v_b,
        d.attn_o_w,
        d.attn_o_b,
        d.attn_norm_gamma,
        d.attn_norm_beta,
        d.ffn_up_w,
        d.ffn_up_b,
        d.ffn_down_w,
        d.ffn_down_b,
        d.ffn_norm_gamma,
        d.ffn_norm_beta};
}

static std::size_t h2d_bytes(const HostData& h) {
    return h.input_ids.bytes() + h.token_type_ids.bytes() + h.attention_mask.bytes() +
           h.hidden_ready.bytes() + h.token_emb.bytes() + h.pos_emb.bytes() +
           h.seg_emb.bytes() + h.emb_gamma.bytes() + h.emb_beta.bytes() +
           h.attn_q_w.bytes() + h.attn_q_b.bytes() + h.attn_k_w.bytes() +
           h.attn_k_b.bytes() + h.attn_v_w.bytes() + h.attn_v_b.bytes() +
           h.attn_o_w.bytes() + h.attn_o_b.bytes() + h.attn_norm_gamma.bytes() +
           h.attn_norm_beta.bytes() + h.ffn_up_w.bytes() + h.ffn_up_b.bytes() +
           h.ffn_down_w.bytes() + h.ffn_down_b.bytes() +
           h.ffn_norm_gamma.bytes() + h.ffn_norm_beta.bytes();
}

static void print_device_info(const cl::Device& device) {
    const cl_ulong global_mem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    std::cout << "\n=== U250 / OpenCL device ===\n"
              << "Name          : " << device.getInfo<CL_DEVICE_NAME>() << '\n'
              << "Vendor        : " << device.getInfo<CL_DEVICE_VENDOR>() << '\n'
              << "Device version: " << device.getInfo<CL_DEVICE_VERSION>() << '\n'
              << "Global memory : " << std::fixed << std::setprecision(2)
              << (static_cast<double>(global_mem) / (1024.0 * 1024.0 * 1024.0)) << " GiB\n";

    const std::string name = lower_copy(device.getInfo<CL_DEVICE_NAME>());
    if (name.find("u250") == std::string::npos) {
        std::cout << "CANH BAO: device name khong chua 'u250'; xclbin co the khong dung platform.\n";
    }
}

static void print_layer_table(const std::vector<LayerTiming>& timings) {
    std::cout << "\n=== Thoi gian tung layer (ms) ===\n";
    std::cout << std::left << std::setw(7) << "Layer"
              << std::right << std::setw(11) << "Wall"
              << std::setw(11) << "Embed"
              << std::setw(11) << "QKV"
              << std::setw(11) << "Attn"
              << std::setw(11) << "AttnOut"
              << std::setw(11) << "FFN-Up"
              << std::setw(11) << "FFN-Down" << '\n';

    for (std::size_t i = 0; i < timings.size(); ++i) {
        const auto& t = timings[i];
        std::cout << std::left << std::setw(7) << i
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(11) << t.wall_ms
                  << std::setw(11) << t.embedding_ms
                  << std::setw(11) << t.qkv_ms
                  << std::setw(11) << t.attn_core_ms
                  << std::setw(11) << t.attn_out_ms
                  << std::setw(11) << t.ffn_up_ms
                  << std::setw(11) << t.ffn_down_ms << '\n';
    }
    std::cout << "Luu y: cac event kernel chay chong lap, khong cong 6 cot kernel de ra Wall.\n";
}

static void verify_and_print_output(const HostData& h) {
    // 12 layer la so chan: layer 11 ghi output vao hidden_ping.
    const float* out = h.hidden_ping.data();
    double checksum = 0.0;
    double l1 = 0.0;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    std::size_t non_finite = 0;

    for (std::size_t i = 0; i < cfg::TOKEN_FLOATS; ++i) {
        const float value = out[i];
        if (!std::isfinite(value)) {
            ++non_finite;
            continue;
        }
        checksum += static_cast<double>(value);
        l1 += std::fabs(static_cast<double>(value));
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }

    std::cout << "\n=== Kiem tra output hidden_ping ===\n";
    std::cout << "First 16 floats: ";
    for (int i = 0; i < 16; ++i) {
        std::cout << std::fixed << std::setprecision(6) << out[i]
                  << ((i == 15) ? '\n' : ' ');
    }
    std::cout << "Checksum      : " << std::setprecision(9) << checksum << '\n'
              << "L1 norm       : " << l1 << '\n'
              << "Min / Max     : " << min_value << " / " << max_value << '\n'
              << "Non-finite    : " << non_finite << " / " << cfg::TOKEN_FLOATS << '\n';

    if (non_finite != 0) {
        throw std::runtime_error("Output co NaN/Inf; can kiem tra kernel hoac mapping bo nho.");
    }
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) {
            std::cerr << "Usage: " << argv[0] << " <bert.xclbin> [so_lan_chay]\n";
            return EXIT_FAILURE;
        }

        const std::string xclbin_path = argv[1];
        const int runs = (argc == 3) ? std::stoi(argv[2]) : 1;
        if (runs <= 0) {
            throw std::invalid_argument("so_lan_chay phai > 0");
        }

        std::cout << "BERT-Base split-kernel host\n"
                  << "SEQ_LEN       : " << cfg::SEQ_LEN << " (active du 128 token)\n"
                  << "HIDDEN        : " << cfg::HIDDEN << '\n'
                  << "FFN_DIM       : " << cfg::FFN_DIM << '\n'
                  << "NUM_LAYERS    : " << cfg::NUM_LAYERS << '\n'
                  << "Benchmark runs: " << runs << '\n';

        const auto init_start = std::chrono::steady_clock::now();
        HostData host;
        initialize_fake_data(host);
        const auto init_end = std::chrono::steady_clock::now();
        const double init_ms =
            std::chrono::duration<double, std::milli>(init_end - init_start).count();
        std::cout << "Khoi tao ~415 MiB du lieu gia: " << std::fixed << std::setprecision(3)
                  << init_ms << " ms\n";

        const auto xclbin = read_binary(xclbin_path);
        double program_ms = 0.0;
        Runtime runtime = program_first_compatible_device(xclbin, program_ms);
        print_device_info(runtime.device);
        std::cout << "Nap/program xclbin: " << std::fixed << std::setprecision(3)
                  << program_ms << " ms\n";

        DeviceBuffers device = create_device_buffers(runtime.context, host);
        Kernels kernels = create_kernels(runtime.program);

        const auto h2d_objects = build_h2d_list(device);
        cl::Event h2d_event;
        const auto h2d_wall_start = std::chrono::steady_clock::now();
        runtime.queue.enqueueMigrateMemObjects(h2d_objects, 0, nullptr, &h2d_event);
        h2d_event.wait();
        const auto h2d_wall_end = std::chrono::steady_clock::now();
        const double h2d_wall_ms =
            std::chrono::duration<double, std::milli>(h2d_wall_end - h2d_wall_start).count();
        const double h2d_device_ms = event_duration_ms(h2d_event);
        const double transferred_mib = static_cast<double>(h2d_bytes(host)) / (1024.0 * 1024.0);
        const double h2d_gib_s = (transferred_mib / 1024.0) / (h2d_wall_ms / 1000.0);

        std::cout << "\n=== Host -> U250 ===\n"
                  << "Transferred    : " << std::fixed << std::setprecision(2)
                  << transferred_mib << " MiB\n"
                  << "H2D wall       : " << std::setprecision(3) << h2d_wall_ms << " ms\n"
                  << "H2D event      : " << h2d_device_ms << " ms\n"
                  << "Effective BW   : " << h2d_gib_s << " GiB/s\n";

        std::vector<double> run_totals_ms;
        std::vector<LayerTiming> last_run_timings;
        run_totals_ms.reserve(static_cast<std::size_t>(runs));

        for (int run = 0; run < runs; ++run) {
            // Token sau lan chay truoc dang bang 12, phai reset ve 0 truoc layer 0.
            host.hidden_ready[0] = 0;
            cl::Event reset_event;
            runtime.queue.enqueueMigrateMemObjects(
                std::vector<cl::Memory>{device.hidden_ready}, 0, nullptr, &reset_event);
            reset_event.wait();

            std::vector<LayerTiming> layer_timings;
            layer_timings.reserve(cfg::NUM_LAYERS);

            const auto run_start = std::chrono::steady_clock::now();
            for (int layer = 0; layer < cfg::NUM_LAYERS; ++layer) {
                layer_timings.push_back(
                    run_one_layer(runtime.queue, kernels, device, layer));
            }
            const auto run_end = std::chrono::steady_clock::now();

            const double total_ms =
                std::chrono::duration<double, std::milli>(run_end - run_start).count();
            run_totals_ms.push_back(total_ms);
            last_run_timings = std::move(layer_timings);

            std::cout << "Run " << run << ": " << std::fixed << std::setprecision(3)
                      << total_ms << " ms, "
                      << (cfg::SEQ_LEN / (total_ms / 1000.0)) << " token/s\n";
        }

        print_layer_table(last_run_timings);

        const double avg_ms =
            std::accumulate(run_totals_ms.begin(), run_totals_ms.end(), 0.0) /
            static_cast<double>(run_totals_ms.size());
        const auto minmax = std::minmax_element(run_totals_ms.begin(), run_totals_ms.end());

        std::cout << "\n=== Tong ket inference 12 layer ===\n"
                  << "Average latency: " << std::fixed << std::setprecision(3) << avg_ms << " ms\n"
                  << "Min latency    : " << *minmax.first << " ms\n"
                  << "Max latency    : " << *minmax.second << " ms\n"
                  << "Throughput     : " << (cfg::SEQ_LEN / (avg_ms / 1000.0)) << " token/s\n";

        // NUM_LAYERS = 12 la so chan, output cuoi nam trong hidden_ping.
        cl::Event d2h_event;
        const auto d2h_start = std::chrono::steady_clock::now();
        runtime.queue.enqueueMigrateMemObjects(
            std::vector<cl::Memory>{device.hidden_ping},
            CL_MIGRATE_MEM_OBJECT_HOST,
            nullptr,
            &d2h_event);
        d2h_event.wait();
        const auto d2h_end = std::chrono::steady_clock::now();
        const double d2h_wall_ms =
            std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();

        std::cout << "D2H output " << (host.hidden_ping.bytes() / 1024.0) << " KiB: "
                  << std::fixed << std::setprecision(3) << d2h_wall_ms
                  << " ms (event " << event_duration_ms(d2h_event) << " ms)\n";

        verify_and_print_output(host);
        std::cout << "\nPASS: card da chay du 128 token x 12 layer va output khong co NaN/Inf.\n";
        return EXIT_SUCCESS;

    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
