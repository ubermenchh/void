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
    double scale = sqrt(in_dim + out_dim);
    linear->weight = tensor_randn(in_dim, out_dim, true, 0);
    linear->weight->data = MatrixScalarDiv(linear->weight->data, scale);

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

    sgd->params = params; // all the parameters of a Module
    sgd->param_count = param_count; // number of parameters
    sgd->lr = lr; // learning rate

    return (Optim*)sgd;
}

void free_sgd(Optim* optim) {
    // frees up the memory 
    SGD* sgd = (SGD*)optim->impl;
    free(sgd);
}

void sgd_step(Optim* optim) {
    SGD* sgd = (SGD*)optim->impl;
    const double clip_value = 1.f;
    for (int i = 0; i < sgd->param_count; i++) {
        Tensor* param = sgd->params[i];
        if (param->grad) {
            for (int j = 0; j < param->data->rows * param->data->cols; j++) {
                double grad = param->grad->data->data[j];
                // data -= lr*grad
                param->data->data[j] -= sgd->lr * grad; 
            }
        }
    }
}

void sgd_zero_grad(Optim* optim) {
    SGD* sgd = (SGD*)optim->impl;
    for (int i = 0; i < sgd->param_count; i++) {
        Tensor* param = sgd->params[i];
        // sets all the elements of grad to 0
        if (param->grad) {
            for (int j = 0; j < param->grad->data->rows*param->grad->data->cols; j++) {
                param->grad->data->data[j] = 0.0;
            }
        } else {
            param->grad = init_tensor(MatrixZerosLike(param->data), false);
        }
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
    //free_tensor(diff);
    return out;
}

void mse_backward(Context* ctx, Tensor* grad_output) {
    Tensor* y_true = ctx->saved_tensors[0];
    Tensor* y_pred = ctx->saved_tensors[1];
    float eps = 1e-8;
    double scale = 2.0 / (y_pred->data->rows * y_pred->data->cols + eps);

    if (y_pred->requires_grad) {
        Matrix* dmse = MatrixScalarMul(MatrixSub(y_true->data, y_pred->data), scale);
        y_pred->grad = tensor_add(y_pred->grad, init_tensor(MatrixMultiply(dmse, grad_output->data), false));
        FreeMatrix(dmse);
    }
}

Tensor* ce_loss(Tensor* output, Tensor* target) {
    Tensor* log_output = tensor_log(output); // takes the log of the predicted probs
    Tensor* neg_log_likelihood = tensor_multiply(log_output, target); // mutlitply with target, so that only the true class probs remain

    Tensor* sum = tensor_sum(neg_log_likelihood, 0);
    Tensor* loss = tensor_mean(sum, -1); // overall mean of the log_likelihood
    Tensor* neg_loss = tensor_negate(loss); // negate it

    Tensor* saved_tensors[2] = {output, target};
    neg_loss->_ctx = init_context(ce_backward, saved_tensors, 2);

    free_tensor(loss);
    free_tensor(neg_log_likelihood);
    free_tensor(log_output);

    return neg_loss;
}

void ce_backward(Context* ctx, Tensor* grad_output) {
    Tensor* output = ctx->saved_tensors[0];
    Tensor* target = ctx->saved_tensors[1];
    
    if (output->requires_grad) {
        Tensor* grad = tensor_sub(output, target);
        output->grad = tensor_multiply(grad, grad_output);

        free_tensor(grad);
    }
}

Tensor* softmax(Tensor* tensor) {
    Tensor* max_vals = tensor_max(tensor, -1);
    Tensor* shifted = tensor_sub(tensor, max_vals);
    Tensor* exp_vals = tensor_exp(shifted);
    Tensor* sum_exp = tensor_sum(exp_vals, -1);
    Tensor* out = tensor_divide(exp_vals, tensor_full_like(exp_vals, sum_exp->data->data[0], false));

    if (out->requires_grad) {
        Tensor* saved_tensors[1] = {out};
        out->_ctx = init_context(softmax_backward, saved_tensors, 1);
    }

    free_tensor(sum_exp);
    free_tensor(exp_vals);
    free_tensor(shifted);
    free_tensor(max_vals);

    return out;
}

void softmax_backward(Context* ctx, Tensor* grad_output) {
    Tensor* softmax_out = ctx->saved_tensors[0];

    if (softmax_out->requires_grad) {
        Tensor* grad = tensor_multiply(softmax_out, grad_output);
        Tensor* sum_grad = tensor_sum(grad, -1);
        Tensor* broadcasted_sum = tensor_broadcast(sum_grad, grad->data->rows, grad->data->cols);
        Tensor* scaled_output = tensor_multiply(softmax_out, broadcasted_sum);
        Tensor* final_grad = tensor_sub(grad, scaled_output);

        softmax_out->grad = final_grad;

        free_tensor(grad);
        free_tensor(sum_grad);
        free_tensor(broadcasted_sum);
        free_tensor(scaled_output);
    }
}

Tensor* softmax_cross_entropy(Tensor* logits, Tensor* target) {
    Tensor* max_logits = tensor_max(logits, 1);
    Tensor* shifted_logits = tensor_sub(logits, max_logits);
    Tensor* exp_logits = tensor_exp(shifted_logits);
    Tensor* sum_exp = tensor_sum(exp_logits, 1);
    Tensor* log_sum_exp = tensor_log(sum_exp);
    
    Tensor* target_logits = tensor_multiply(logits, target);
    Tensor* target_sum = tensor_sum(target_logits, 1);
    
    Tensor* loss = tensor_sub(log_sum_exp, target_sum);
    Tensor* mean_loss = tensor_mean(loss, -1);
    
    // Set up backward pass
    Tensor* saved_tensors[2] = {exp_logits, target};
    mean_loss->_ctx = init_context(softmax_cross_entropy_backward, saved_tensors, 2);
    
    // Clean up
    free_tensor(max_logits);
    free_tensor(shifted_logits);
    //free_tensor(exp_logits);
    //free_tensor(sum_exp);
    free_tensor(log_sum_exp);
    free_tensor(target_logits);
    free_tensor(target_sum);
    free_tensor(loss);
    
    return mean_loss;
}

void softmax_cross_entropy_backward(Context* ctx, Tensor* grad_output) {
    Tensor* exp_logits = ctx->saved_tensors[0];
    Tensor* target = ctx->saved_tensors[1];
    
    Tensor* sum_exp = tensor_sum(exp_logits, 1);
    Tensor* softmax_output = tensor_divide(exp_logits, 
            tensor_full_like(exp_logits, sum_exp->data->data[0], false));

    Tensor* grad = tensor_sub(softmax_output, target);
    float scale = grad_output->data->data[0] / exp_logits->data->rows;
    Tensor* final_grad = tensor_scale(grad, scale);
    
    exp_logits->grad = final_grad;
    
    // Clean up
    //free_tensor(sum_exp);
    free_tensor(softmax_output);
    free_tensor(grad);
}
