#include "void.h" 

Tensor** module_parameters(Module* module, int* count) {
    if (module->derived && module->parameters) {
        return module_parameters(module, count);
    }
    *count = 0;
    return NULL;
}

Linear* init_linear(int in_dim, int out_dim, bool has_bias) {
    // Initializes a Linear Layer 
    Linear* linear = (Linear*)malloc(sizeof(Linear));
    linear->weight = tensor_randn(in_dim, out_dim, true, 0);
    if (has_bias) {
        linear->bias = tensor_zeros(1, out_dim, true);
    } else {
        linear->bias = NULL;
    }
    linear->has_bias = has_bias;

    linear->base.forward = linear_forward;
    linear->base.parameters = linear_parameters;
    linear->base.derived = linear;

    return linear;
}

void free_linear(Linear* linear) {
    // Frees an already Initialized linear layer
    free_tensor(linear->weight);
    if (linear->has_bias) free_tensor(linear->bias);
    free(linear);
}

Tensor* linear_forward(Module* module, Tensor* input) {
    Linear* linear = (Linear*)module->derived;
    Tensor* out = tensor_matmul(input, linear->weight);
    if (linear->has_bias) out = tensor_add(out, linear->bias);
    return out;
}

Tensor** linear_parameters(Module* module, int* count) {
    Linear* linear = (Linear*)module->derived;
    Tensor** params = (Tensor**)malloc(sizeof(Tensor*) * 2);
    int idx = 0;
    params[idx++] = linear->weight;
    if (linear->has_bias) params[idx++] = linear->bias;
    *count = idx;
    return params;
}

SGD* init_sgd(Tensor** params, int param_count, double lr) {
    SGD* sgd = (SGD*)malloc(sizeof(SGD));
    sgd->params = params;
    sgd->param_count = param_count;
    sgd->lr = lr;

    sgd->base.zero_grad = sgd_zero_grad;
    sgd->base.step = sgd_step;
    sgd->base.dervied = sgd;
    return sgd;
}

void free_sgd(SGD* sgd) {
    free(sgd);
}

void sgd_step(void* optim) {
    SGD* sgd = (SGD*)optim;
    for (int i = 0; i < sgd->param_count; i++) {
        Tensor* param = sgd->params[i];
        for (int j = 0; j < param->data->rows * param->data->cols; j++) {
            param->data->data[j] -= sgd->lr * param->grad->data->data[j]; 
        }
    }
}

void sgd_zero_grad(void* optim) {
    SGD* sgd = (SGD*)optim;
    for (int i = 0; i < sgd->param_count; i++) {
        Tensor* param = sgd->params[i];
        param->grad->data = MatrixZerosLike(param->data);
    }
}

Tensor* mse(Tensor* y_true, Tensor* y_pred) {
    Tensor* diff = tensor_sub(y_true, y_pred);
    Tensor* sqr = tensor_pow(diff, 2.0);
    Tensor* out = tensor_mean(sqr, -1); // -1 for overall mean 
    
    if (out->requires_grad) {
        Tensor* saved_tensors[2] = {y_true, y_pred};
        out->_ctx = init_context(mse_backward, saved_tensors, 2);
    }
    free_tensor(sqr);
    free_tensor(diff);
    return out;
}

void mse_backward(Context* ctx, Tensor* grad_output) {
    Tensor* y_true = ctx->saved_tensors[0];
    Tensor* y_pred = ctx->saved_tensors[1];
    double scale = 2.0 / y_pred->data->rows*y_pred->data->cols;

    if (y_pred->requires_grad) {
        Matrix* dmse = MatrixScalarMul(MatrixSub(y_true->data, y_pred->data), scale);
        y_pred->grad = init_tensor(MatrixMultiply(dmse, grad_output->data), false);
        FreeMatrix(dmse);
    }
}
