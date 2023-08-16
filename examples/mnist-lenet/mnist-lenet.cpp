#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <fstream>

#include "ggml.h"

// bswap
#ifdef _MSC_VER
#include <stdlib.h>
#define bswap_32(x) _byteswap_ulong(x)
#elif defined(__APPLE__)
#include <libkern/OSByteOrder.h>
#define bswap_32(x) OSSwapInt32(x)
#else
#include <byteswap.h>
#endif


class model {
    ggml_context* m_ctx;

    ggml_tensor* m_input;

    std::vector<ggml_tensor*> m_weights;

    ggml_tensor* m_output;

    ggml_cgraph m_compute_graph;

    template <int N>
    ggml_tensor* create_tensor(const char* name, const int64_t(&dims)[N]) {
        auto ret = ggml_new_tensor(m_ctx, GGML_TYPE_F32, N, dims);
        if (!ret) {
            throw std::runtime_error("ggml_new_tensor() failed");
        }
        ggml_set_name(ret, name);
        return ret;
    }

    template <int N>
    ggml_tensor* create_weight_tensor(const char* name, const int64_t(&dims)[N]) {
        auto ret = create_tensor(name, dims);
        m_weights.push_back(ret);
        return ret;
    }

public:
    ggml_tensor* tmp;
    model() {
        ggml_init_params params = { /*size*/ 3 * 1024 * 1024,};
        m_ctx = ggml_init(params);
        if (!m_ctx) {
            throw std::runtime_error("ggml_init() failed");
        }

        m_input = create_tensor("input", {28, 28, 1});
        ggml_tensor* next;

        auto conv0_weight = create_weight_tensor("conv0_weight", {5, 5, 1, 4});
        auto conv0_bias = create_weight_tensor("conv0_bias", {1, 1, 4});
        next = ggml_conv_2d(m_ctx, conv0_weight, m_input, 1, 1, 0, 0, 1, 1);
        tmp = next;
        next = ggml_add(m_ctx, next, ggml_repeat(m_ctx, conv0_bias, next));

        next = ggml_tanh(m_ctx, next);

        next = ggml_pool_2d(m_ctx, next, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);

        auto conv1_weight = create_weight_tensor("conv1_weight", {5, 5, 4, 12});
        auto conv1_bias = create_weight_tensor("conv1_bias", {1, 1, 12});
        next = ggml_conv_2d(m_ctx, conv1_weight, next, 1, 1, 0, 0, 1, 1);
        next = ggml_add(m_ctx, next, ggml_repeat(m_ctx, conv1_bias, next));

        next = ggml_tanh(m_ctx, next);

        next = ggml_pool_2d(m_ctx, next, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);

        next = ggml_reshape_1d(m_ctx, next, 12 * 4 * 4);

        auto linear_weight = create_weight_tensor("linear_weight", {12 * 4 * 4, 10});
        auto linear_bias = create_weight_tensor("linear_bias", {10});
        next = ggml_mul_mat(m_ctx, linear_weight, next);
        next = ggml_add(m_ctx, next, linear_bias);

        m_output = ggml_soft_max(m_ctx, next);
        ggml_set_name(m_output, "output");

        m_compute_graph = ggml_build_forward(m_output);
    }

    ~model() {
        ggml_free(m_ctx);
    }

    void load_from_dir(std::string dir) {
        for (auto* w : m_weights) {
            auto path = dir + '/' + w->name + "-f32.raw";
            std::ifstream fin(path, std::ios::in | std::ios::binary);
            fin.read(reinterpret_cast<char*>(w->data), ggml_nbytes(w));
            if (fin.bad()) {
                std::string err = "error reading ";
                err += path;
                throw std::runtime_error(err);
            }
        }
    }

    struct output_elem {
        int digit;
        float prob;
    };
    struct output {
        // 10 elements, sorted by prob, first one being the most probable
        output_elem sorted_elems[10];
    };

    output compute(const float* input) {
        memcpy(m_input->data, input, ggml_nbytes(m_input));
        ggml_graph_compute_with_ctx(m_ctx, &m_compute_graph, 1);
        output ret;
        float* output_data = ggml_get_data_f32(m_output);
        for (int i = 0; i < 10; ++i) {
            ret.sorted_elems[i].digit = i;
            ret.sorted_elems[i].prob = output_data[i];
        }
        std::sort(std::begin(ret.sorted_elems), std::end(ret.sorted_elems),
                       [](const output_elem& a, const output_elem& b) { return a.prob > b.prob; });
        return ret;
    }
};

class idx3_ubyte {
    std::vector<uint8_t> m_buf;
    uint32_t m_num_images;
public:
    idx3_ubyte(const char* path) {
        std::ifstream fin(path, std::ios::in | std::ios::binary);
        uint32_t header[4];
        fin.read((char*)header, 16);
        for (auto& i : header) {
            i = bswap_32(i);
        }
        m_num_images = header[1];
        auto xsize = header[2];
        auto ysize = header[3];

        if (xsize != ysize || xsize != 28) {
            throw std::runtime_error("bad header of idx3_ubyte");
        }

        m_buf.resize(m_num_images * xsize * ysize);
        fin.read((char*)m_buf.data(), m_buf.size());
        if (fin.bad()) {
            throw std::runtime_error("bad data in idx3_ubyte");
        }
    }

    const uint8_t* get_image(uint32_t index) const {
        if (index >= m_num_images) return nullptr;
        return m_buf.data() + index * 28 * 28;
    }

    // also return printed
    const uint8_t* print_image(std::ostream& out, uint32_t index) const {
        auto p = get_image(index);
        if (!p) {
            out << "error\n";
            return nullptr;
        }

        for (uint32_t y = 0; y < 28; ++y) {
            for (uint32_t x = 0; x < 28; ++x) {
                auto val = *p;
                if (val < 10) out << ' ';
                else if (val < 50) out << '-';
                else if (val < 100) out << '+';
                else if (val < 150) out << 'O';
                else if (val < 200) out << '&';
                else out << '@';
                ++p;
            }
            out << '\n';
        }

        return p;
    }
};

int main() {
    try {
        model m;
        m.load_from_dir("C:/prj/build/ggml/examples/mnist-lenet/models");

        idx3_ubyte t10k("C:/prj/build/ggml/examples/mnist/models/mnist/t10k-images.idx3-ubyte");

        auto img = t10k.print_image(std::cout, 9000);
        float input[28 * 28];
        for (auto& i : input) {
            i = float(*img++) / 255;
        }

        auto out = m.compute(input);

        for (auto& o : out.sorted_elems) {
            std::cout << o.digit << ": " << o.prob << '\n';
        }
    }
    catch (std::exception ex) {
        std::cout << ex.what() << '\n';
    }

    return 0;
}
