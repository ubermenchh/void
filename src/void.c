#include "void.h"

Tensor* init_tensor(Matrix* data, bool requires_grad) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    tensor->data = MatrixCopy(data);
    tensor->grad = NULL;
    tensor->requires_grad = requires_grad;
    tensor->grad_fn = NULL;
    tensor->ctx = NULL;

    return tensor;
}

void free_tensor(Tensor* t) {
    FreeMatrix(t->data);
    if (t->grad != NULL) {
        FreeMatrix(t->grad);
    }
    free(t);
}

void save_for_backward(Context* ctx, Tensor** tensors, int num_saved) {
    ctx->saved_tensors = (Tensor**)malloc(num_saved * sizeof(Tensor*));
    if (ctx->saved_tensors == NULL) {
        printf("Failed to allocate memory for saved tensors.\n");
        exit(1);
    }
    
    for (int i = 0; i < num_saved; i++) {
        ctx->saved_tensors[i] = tensors[i];
    }
    ctx->num_saved = num_saved;
}

void print_tensor(Tensor* t) {
    int max_digits = 0;
    double max_val = 0.0;
    double min_val = 0.0;

    for (int i = 0; i < t->data->rows; i++) {
        for (int j = 0; j < t->data->cols; j++) {
            double val = fabs(MAT_AT(t->data, i, j));
            if (val > max_val) {
                max_val = val;
            }
            if (val < min_val || (min_val == 0.0)) {
                min_val = val;
            }
        }
    }

    if ((max_val == 0.0 && min_val == 0.0) || (max_val == 1.0 && min_val == 1.0)) {
        max_digits = 1;
    } else {
        max_digits = (int)log10(max_val) + 1 + 8;
    }

    printf("Tensor(data=(\n[");
    for (int i = 0; i < t->data->rows; i++) {
        if (i == 0) {
            printf("[");
        } else {
            printf(" [");
        }
        for (int j = 0; j < t->data->cols; j++) {
            printf("%*.*f ", max_digits, 5, MAT_AT(t->data, i, j));
        }
        if (i != t->data->rows-1) {
            printf(" ]\n");
        } else {
            printf(" ]");
        }
    }
    printf("]\n");
    printf("), requires_grad=%d)\n\n", t->requires_grad);
}

void print_tensor_grad(Tensor* t) {
    if (t->grad == NULL) {
        printf("Tensor(grad=NULL)\n");
        return;
    }

    int max_digits = 0;
    double max_val = 0.0;
    double min_val = 0.0;

    for (int i = 0; i < t->grad->rows; i++) {
        for (int j = 0; j < t->grad->cols; j++) {
            double val = fabs(MAT_AT(t->grad, i, j));
            if (val > max_val) {
                max_val = val;
            }
            if (val < min_val || (min_val == 0.0)) {
                min_val = val;
            }
        }
    }

    if ((max_val == 0.0 && min_val == 0.0) || (max_val == 1.0 && min_val == 1.0)) {
        max_digits = 1;
    } else {
        max_digits = (int)log10(max_val) + 1 + 8;
    }
    
    if (t->requires_grad) {
        printf("Tensor(grad=(\n[");
        for (int i = 0; i < t->grad->rows; i++) {
            if (i == 0) {
                printf("[");
            } else {
                printf(" [");
            }
            for (int j = 0; j < t->grad->cols; j++) {
                printf("%*.*f ", max_digits, 5, MAT_AT(t->grad, i, j));
            }
            if (i != t->grad->rows-1) {
                printf(" ]\n");
            } else {
                printf(" ]");
            }
        }
        printf("]\n");
        printf(")\n");

    } else {
        printf("Tensor(grad=NULL)\n");
    } 
}

void tensor_backward(Tensor* t, Matrix* grad) {
    if (grad == NULL) {
        grad = MatrixOnesLike(t->data);
    }
    if (t->grad == NULL) {
        t->grad = MatrixCopy(grad);
    } else {
        Matrix* temp = MatrixAdd(t->grad, grad);
        FreeMatrix(t->grad);
        t->grad = temp;
    }
    if (t->grad_fn != NULL) {
        Tensor grad_output;
        grad_output.data = t->grad;
        t->grad_fn(&grad_output, t);
    }
}

Tensor* add(Tensor* a, Tensor* b) {
    Tensor* out = init_tensor(MatrixAdd(a->data, b->data), a->requires_grad || b->requires_grad);

    if (out->requires_grad) {
        out->ctx = INIT_CONTEXT;
        if (out->ctx == NULL) {
            printf("Failed to allocate memory for context.\n");
            return NULL;
        }

        Tensor* saved_tensors[2] = {a, b};
        save_for_backward(out->ctx, saved_tensors, 2);
        out->grad_fn = add_backward;
    }
    return out;
}

void add_backward(Tensor* grad_output, Tensor* out) {
    Tensor* a = out->ctx->saved_tensors[0];
    Tensor* b = out->ctx->saved_tensors[1];

    if (a->requires_grad) {
        tensor_backward(a, grad_output->data);
    } 
    if (b->requires_grad) {
        tensor_backward(b, grad_output->data);
    }
    free(out->ctx->saved_tensors);
    free(out->ctx);
    out->ctx = NULL;
}

Tensor* multiply(Tensor* a, Tensor* b) {
    Tensor* out = init_tensor(
        MatrixMultiply(a->data, b->data), a->requires_grad || b->requires_grad
    );

    if (out->requires_grad) {
        out->ctx = INIT_CONTEXT;
        if (out->ctx == NULL) {
            printf("Failed to allocate memory for context.\n");
            return NULL;
        }

        Tensor* saved_tensors[2] = {a, b};
        save_for_backward(out->ctx, saved_tensors, 2);
        out->grad_fn = mul_backward;
    } 
    return out;
}

void mul_backward(Tensor* grad_output, Tensor* out) {
    Tensor* a = out->ctx->saved_tensors[0];
    Tensor* b = out->ctx->saved_tensors[1];

    if (a->requires_grad) {
        Matrix* grad_a = MatrixMultiply(grad_output->data, b->data);
        tensor_backward(a, grad_a);
        FreeMatrix(grad_a);
    }
    if (b->requires_grad) {
        Matrix* grad_b = MatrixMultiply(grad_output->data, a->data);
        tensor_backward(b, grad_b);
        FreeMatrix(grad_b);
    }
    free(out->ctx->saved_tensors);
    free(out->ctx);
    out->ctx = NULL;
}

Tensor* sum(Tensor* a) {
    Matrix* sum_matrix = InitMatrix(1, 1);
    sum_matrix->data[0] = MatrixSum(a->data);

    Tensor* output = init_tensor(sum_matrix, a->requires_grad);

    if (output->requires_grad) {
        output->grad_fn = sum_backward;
        output->ctx = INIT_CONTEXT; 
        Tensor* saved_tensors[1] = {a};
        save_for_backward(output->ctx, saved_tensors, 1);
    }

    FreeMatrix(sum_matrix);
    return output;
}

void sum_backward(Tensor* grad_output, Tensor* out) {
    Context* ctx = out->ctx;
    Tensor* input = ctx->saved_tensors[0];

    if (input->requires_grad) {
        Matrix* grad_input = MatrixOnesLike(input->data);
        
        for (int i = 0; i < grad_input->rows * grad_input->cols; i++) {
            grad_input->data[i] *= grad_output->data->data[0];
        }

        tensor_backward(input, grad_input);
        FreeMatrix(grad_input);
    }
    free(ctx->saved_tensors);
    free(ctx);
    out->ctx = NULL;
}

Tensor* negate(Tensor* input) {
    Tensor* out = init_tensor(MatrixNeg(input->data), input->requires_grad);

    if (out->requires_grad) {
        out->ctx = INIT_CONTEXT; 
        out->grad_fn = neg_backward;
        Tensor* saved_tensors[1] = {input};
        save_for_backward(out->ctx, saved_tensors, 1);
    }
    return out;
}

void neg_backward(Tensor* grad_output, Tensor* out) {
    Tensor* input = out->ctx->saved_tensors[0];

    if (input->requires_grad) {
        Matrix* grad_input = MatrixOnesLike(input->data);

        for (int i = 0; i < grad_input->rows * grad_input->cols; i++) {
            grad_input->data[i] *= grad_output->data->data[0];
        }

        tensor_backward(input, grad_input);
        FreeMatrix(grad_input);
    }
    free(out->ctx->saved_tensors);
    free(out->ctx);
    out->ctx = NULL;
}

Tensor* divide(Tensor* a, Tensor* b) {
    Tensor* out = init_tensor(MatrixDivide(a->data, b->data), a->requires_grad || b->requires_grad);

    if (out->requires_grad) {
        out->ctx = INIT_CONTEXT;
        out->grad_fn = div_backward;
        Tensor* saved_tensors[2] = {a, b};
        save_for_backward(out->ctx, saved_tensors, 2);
    }
    return out;
}

void div_backward(Tensor* grad_output, Tensor* out) {
    Tensor* a = out->ctx->saved_tensors[0];
    Tensor* b = out->ctx->saved_tensors[1];

    if (a->requires_grad) {
        Matrix* grad_a = MatrixDivide(grad_output->data, b->data);
        tensor_backward(a, grad_a);
        FreeMatrix(grad_a);
    }
    if (b->requires_grad) {
        Matrix* b_sq = MatrixMultiply(b->data, b->data);
        Matrix* temp = MatrixMultiply(grad_output->data, a->data);
        Matrix* grad_b = MatrixNeg(MatrixDivide(temp, b_sq));

        tensor_backward(b, grad_b);

        FreeMatrix(b_sq);
        FreeMatrix(temp);
        FreeMatrix(grad_b);
    }
    free(out->ctx->saved_tensors);
    free(out->ctx);
    out->ctx = NULL;
}
