#include "void.h" 

#define SEED 1337

Module* init_linear(int in_dim, int out_dim, bool has_bias) {
    // Initializes a Linear Layer 
    Linear* linear = (Linear*)malloc(sizeof(Linear));
    linear->base.impl = linear;
    linear->base.forward = linear_forward;
    linear->base.parameters = linear_parameters;
    linear->base.free = free_linear;

    linear->in_dim = in_dim;
    linear->out_dim = out_dim;
    linear->has_bias = has_bias;

    // 0 means no seed, replace 0 with SEED if you want
    linear->weight = tensor_randn(in_dim, out_dim, true, 0);
    linear->weight->data = MatrixScalarDiv(linear->weight->data, sqrt(in_dim + out_dim));

    linear->bias = has_bias ? tensor_zeros(1, out_dim, true) : NULL;

    return (Module*)linear;
}

void free_linear(Module* module) {
    // Frees an already Initialized linear layer
    Linear* linear = (Linear*)module->impl;
    free_tensor(linear->weight);
    if (linear->has_bias) free_tensor(linear->bias);
    free(linear);
}

Tensor* linear_forward(Module* module, Tensor* input) {
    Linear* linear = (Linear*)module->impl;
    Tensor* out = tensor_matmul(input, linear->weight);
    if (linear->has_bias) out = tensor_add(out, linear->bias);
    return out;
}

Tensor** linear_parameters(Module* module, int* count) {
    Linear* linear = (Linear*)module->impl;
    *count = linear->has_bias ? 2 : 1;
    Tensor** params = malloc(*count * sizeof(Tensor));
    params[0] = linear->weight;
    if (linear->has_bias) params[1] = linear->bias;
    return params;
}

Optim* init_sgd(Tensor** params, int param_count, double lr) {
    SGD* sgd = malloc(sizeof(SGD));
    sgd->base.impl = sgd;
    sgd->base.zero_grad = sgd_zero_grad;
    sgd->base.step = sgd_step;
    sgd->base.free = free_sgd;

    sgd->params = params;
    sgd->param_count = param_count;
    sgd->lr = lr;

    return (Optim*)sgd;
}

void free_sgd(Optim* optim) {
    SGD* sgd = (SGD*)optim->impl;
    free(sgd);
}

void sgd_step(Optim* optim) {
    SGD* sgd = (SGD*)optim->impl;
    for (int i = 0; i < sgd->param_count; i++) {
        Tensor* param = sgd->params[i];
        for (int j = 0; j < param->data->rows * param->data->cols; j++) {
            param->data->data[j] += (sgd->lr * param->grad->data->data[j]); 
        }
    }
}

void sgd_zero_grad(Optim* optim) {
    SGD* sgd = (SGD*)optim->impl;
    for (int i = 0; i < sgd->param_count; i++) {
        Tensor* param = sgd->params[i];
        param->grad = init_tensor(MatrixZerosLike(param->data), false);
        //for (int j = 0; j < param->data->rows * param->data->cols; j++) {
        //    param->grad->data->data[j] = 0.0;
        //}
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

Tensor* ce_loss(Tensor* output, Tensor* target) {
    Tensor* log_output = tensor_log(output);
    Tensor* neg_log_likelihood = tensor_multiply(target, log_output);
    
    Tensor* loss = tensor_mean(neg_log_likelihood, -1);
    Tensor* neg_loss = tensor_negate(loss);


    free_tensor(loss);
    free_tensor(neg_log_likelihood);
    free_tensor(log_output);

    return neg_loss;
}

Tensor* softmax(Tensor* tensor) {
    Tensor* max_vals = tensor_max(tensor, -1);
    Tensor* shifted = tensor_sub(tensor, max_vals);
    Tensor* exp_vals = tensor_exp(shifted);
    Tensor* sum_exp = tensor_sum(exp_vals);
    Tensor* out = tensor_divide(exp_vals, tensor_full_like(exp_vals, sum_exp->data->data[0], false));

    free_tensor(sum_exp);
    free_tensor(exp_vals);
    free_tensor(shifted);
    free_tensor(max_vals);

    return out;
}
