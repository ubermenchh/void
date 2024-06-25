#include "void.h" 

Linear* init_linear(int in_dim, int out_dim, bool has_bias) {
    // Initializes a Linear Layer
    Linear* lin = (Linear*)malloc(sizeof(Linear));
    lin->in_dim = in_dim;
    lin->out_dim = out_dim;
    lin->has_bias = has_bias;

    lin->weight = tensor_randn(in_dim, out_dim, true, 0);
    if (has_bias) {
        lin->bias = tensor_zeros(1, out_dim, true);
    } else {
        lin->bias = NULL;
    }
    return lin;
}

void free_linear(Linear* lin) {
    // Frees an already Initialized linear layer
    free_tensor(lin->weight);
    free_tensor(lin->bias);
    free(lin);
}

Tensor* linear_forward(Linear* lin, Tensor* input) {
    // out = X@W.T + B
    Tensor* out = tensor_matmul(input, lin->weight);

    if (lin->has_bias) {
        Tensor* biased = tensor_add(out, lin->bias);
        free_tensor(out);
        out = biased;
    }
    return out;
}

LayerNorm* init_layernorm(int n_embed) {
    LayerNorm* ln = (LayerNorm*)malloc(sizeof(LayerNorm));
    
    ln->gamma = tensor_ones(1, n_embed, true);
    ln->beta = tensor_zeros(1, n_embed, true);

    return ln;
}

void free_layernorm(LayerNorm* ln) {
    free_tensor(ln->gamma);
    free_tensor(ln->beta);
    free(ln);
}

Tensor* layernorm_forward(LayerNorm* ln, Tensor* input) {
    Tensor* std_input = tensor_std(input, 1);
    Tensor* mean_input = tensor_mean(input, 1);
    Tensor* norm_input = tensor_divide(tensor_sub(input, mean_input), std_input);

    free_tensor(std_input);
    free_tensor(mean_input);

    return norm_input;
}

Dropout* init_dropout(double drop_prob) {
    Dropout* dp = (Dropout*)malloc(sizeof(Dropout));
    
    dp->drop_prob = drop_prob;
    dp->train_mode = true;
    dp->mask = NULL;

    return dp;
}

void free_dropout(Dropout* dp) {
    if (dp->mask) {
        free_tensor(dp->mask);
    }
    free(dp);
}

Tensor* dropout_forward(Dropout* dp, Tensor* input) {
    if (!dp->train_mode) {
        return input;
    }
    if (dp->mask) {
        free_tensor(dp->mask);
    }

    dp->mask = tensor_mask(input->data->rows, input->data->cols, dp->drop_prob, false);
    print_tensor(dp->mask);
    Tensor* out = tensor_multiply(input, dp->mask);

    return out; 
}

Tensor* mse(Tensor* y_true, Tensor* y_pred) {
    Tensor* diff = tensor_sub(y_true, y_pred);
    Tensor* sqr = tensor_pow(diff, 2.0);
    Tensor* out = tensor_mean(sqr, -1); // -1 for overall mean 
    
    if (out->requires_grad) {
        out->ctx = init_context();
        Tensor* saved_tensors[3] = {out, y_true, y_pred};
        save_for_backward(out->ctx, saved_tensors, 1, NULL, 0);
        out->grad_fn = mse_backward;
    }

    free_tensor(sqr);
    free_tensor(diff);

    return out;
}

void mse_backward(Tensor* grad_output, Tensor* out) {
    Tensor* output = out->ctx->saved_tensors[0];
    Tensor* y_true = out->ctx->saved_tensors[1];
    Tensor* y_pred = out->ctx->saved_tensors[2];

    if (output->requires_grad) {
        Matrix* dmse = MatrixScalarMul(MatrixSub(y_true->data, y_pred->data), 2);
        Matrix* grad_input = MatrixMultiply(dmse, grad_output->data);
        
        FreeMatrix(dmse);
        tensor_backward(output, grad_input);
        FreeMatrix(grad_input);
    }
    free_tensor(y_true);
    free_tensor(y_pred);
    free_context(out->ctx);
    out->ctx = NULL;
}
