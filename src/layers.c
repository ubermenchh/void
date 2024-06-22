#include "void.h" 

Linear* init_linear(int in_dim, int out_dim, bool has_bias) {
    Linear* lin = (Linear*)malloc(sizeof(Linear));
    lin->in_dim = in_dim;
    lin->out_dim = out_dim;
    lin->has_bias = has_bias;

    lin->weight = init_tensor(RandMatrix(in_dim, out_dim, 0), true);
    if (has_bias) {
        lin->bias = init_tensor(ZerosMatrix(1, out_dim), true);
    } else {
        lin->bias = NULL;
    }
    return lin;
}

void free_linear(Linear* lin) {
    free_tensor(lin->weight);
    free_tensor(lin->bias);
    free(lin);
}

Tensor* linear_forward(Linear* lin, Tensor* input) {
    Tensor* out = tensor_matmul(input, lin->weight);

    if (lin->has_bias) {
        Tensor* biased = tensor_add(out, lin->bias);
        free_tensor(out);
        out = biased;
    }
    return out;
}

