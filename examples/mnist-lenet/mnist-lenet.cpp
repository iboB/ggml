#include <iostream>
#include <stdexcept>
#include <algorithm>

#include "ggml.h"

class model {
    ggml_context* m_ctx;

    ggml_tensor* m_input;

    ggml_tensor* m_conv0_weight;
    ggml_tensor* m_conv0_bias;
    ggml_tensor* m_conv1_weight;
    ggml_tensor* m_conv1_bias;
    ggml_tensor* m_linear_weight;
    ggml_tensor* m_linear_bias;

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

public:
    model() {
        ggml_init_params params = { /*size*/ 3 * 1024 * 1024,};
        m_ctx = ggml_init(params);
        if (!m_ctx) {
            throw std::runtime_error("ggml_init() failed");
        }

        m_input = create_tensor("input", {28, 28, 1});
        ggml_tensor* next;

        m_conv0_weight = create_tensor("conv0_weight", {5, 5, 1, 4});
        m_conv0_bias = create_tensor("conv0_bias", {1, 1, 4});
        next = ggml_conv_2d(m_ctx, m_conv0_weight, m_input, 1, 1, 0, 0, 1, 1);
        next = ggml_add(m_ctx, next, ggml_repeat(m_ctx, m_conv0_bias, next));

        next = ggml_tanh(m_ctx, next);

        next = ggml_pool_2d(m_ctx, next, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);

        m_conv1_weight = create_tensor("conv1_weight", {5, 5, 4, 12});
        m_conv1_bias = create_tensor("conv1_bias", {1, 1, 12});
        next = ggml_conv_2d(m_ctx, m_conv1_weight, next, 1, 1, 0, 0, 1, 1);
        next = ggml_add(m_ctx, next, ggml_repeat(m_ctx, m_conv1_bias, next));

        next = ggml_tanh(m_ctx, next);

        next = ggml_pool_2d(m_ctx, next, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);

        next = ggml_reshape_1d(m_ctx, next, 12 * 4 * 4);

        m_linear_weight = create_tensor("linear_weight", {12 * 4 * 4, 10});
        m_linear_bias = create_tensor("linear_bias", {10});
        next = ggml_mul_mat(m_ctx, m_linear_weight, next);
        next = ggml_add(m_ctx, next, m_linear_bias);

        m_output = ggml_soft_max(m_ctx, next);
        ggml_set_name(m_output, "output");

        m_compute_graph = ggml_build_forward(m_output);
    }

    ~model() {
        ggml_free(m_ctx);
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

int main() {
    model m;
    return 0;
}
