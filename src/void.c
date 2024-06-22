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

Context* init_context() {
    Context* ctx = (Context*)malloc(sizeof(Context));
    ctx->saved_tensors = NULL;
    ctx->num_saved_tensors = 0;
    ctx->saved_scalars = NULL;
    ctx->num_saved_scalars = 0;
    return ctx;
}

void free_context(Context* ctx) {
    if (ctx) {
        free(ctx->saved_tensors);
        free(ctx->saved_scalars);
        free(ctx);
    }
}

void save_for_backward(Context* ctx, Tensor** tensors, int num_tensors, double* scalars, int num_scalars) {
    if (num_tensors > 0) {
        ctx->saved_tensors = (Tensor**)malloc(num_tensors * sizeof(Tensor*)); 
        for (int i = 0; i < num_tensors; i++) {
            ctx->saved_tensors[i] = tensors[i];
        }
        ctx->num_saved_tensors = num_tensors;
    }
    if (num_scalars > 0) {
        ctx->saved_scalars = (double*)malloc(num_scalars * sizeof(double));
        for (int i = 0; i < num_scalars; i++) {
            ctx->saved_scalars[i] = scalars[i];
        }
        ctx->num_saved_scalars = num_scalars;
    }
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

void tensor_shape(Tensor* t) {
    printf("Shape: (%d, %d)\n", t->data->rows, t->data->cols);
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

Tensor* tensor_rand(int rows, int cols, bool requires_grad, int seed) {
    return init_tensor(RandMatrix(rows, cols, seed), requires_grad);
}

Tensor* tensor_randn(int rows, int cols, bool requires_grad, int seed) {
    return init_tensor(RandnMatrix(rows, cols, seed), requires_grad);
}

Tensor* tensor_ones(int rows, int cols, bool requires_grad) {
    return init_tensor(OnesMatrix(rows, cols), requires_grad);
}

Tensor* tensor_zeros(int rows, int cols, bool requires_grad) {
    return init_tensor(ZerosMatrix(rows, cols), requires_grad);
}

Tensor* tensor_eye(int size, bool requires_grad) {
    return init_tensor(IdentityMatrix(size), requires_grad);
}

Tensor* tensor_ones_like(Tensor* t, bool requires_grad) {
    return init_tensor(MatrixOnesLike(t->data), requires_grad);
}

Tensor* tensor_zeros_like(Tensor* t, bool requires_grad) {
    return init_tensor(MatrixZerosLike(t->data), requires_grad);
}

Tensor* tensor_full(int rows, int cols, double value, bool requires_grad) {
    return init_tensor(MatrixFull(rows, cols, value), requires_grad);
}

Tensor* tensor_full_like(Tensor* t, double value, bool requires_grad) {
    return init_tensor(MatrixFullLike(t->data, value), requires_grad);
}
    
Tensor* tensor_mask(int rows, int cols, double prob, bool requires_grad) {
    return init_tensor(MatrixMask(rows, cols, prob), requires_grad);
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    // out = a + b
    Tensor* out = init_tensor(MatrixAdd(a->data, b->data), a->requires_grad || b->requires_grad);

    if (out->requires_grad) {
        out->ctx = init_context();
        if (out->ctx == NULL) {
            printf("Failed to allocate memory for context.\n");
            return NULL;
        }

        Tensor* saved_tensors[2] = {a, b};
        save_for_backward(out->ctx, saved_tensors, 2, NULL, 0);
        out->grad_fn = add_backward;
    }
    return out;
}

void add_backward(Tensor* grad_output, Tensor* out) {
    Tensor* a = out->ctx->saved_tensors[0];
    Tensor* b = out->ctx->saved_tensors[1];

    if (a->requires_grad) {
        // grad_a = 1 * grad_output
        tensor_backward(a, grad_output->data);
    } 
    if (b->requires_grad) {
        // grad_b = 1 * grad_output
        tensor_backward(b, grad_output->data);
    }
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_sub(Tensor* a, Tensor* b) {
    Tensor* neg_b = tensor_negate(b);
    Tensor* out = tensor_add(a, neg_b);

    free_tensor(neg_b);
    return out;
}

Tensor* tensor_multiply(Tensor* a, Tensor* b) {
    // out = a*b
    Tensor* out = init_tensor(
        MatrixMultiply(a->data, b->data), a->requires_grad || b->requires_grad
    );

    if (out->requires_grad) {
        out->ctx = init_context();
        if (out->ctx == NULL) {
            printf("Failed to allocate memory for context.\n");
            return NULL;
        }

        Tensor* saved_tensors[2] = {a, b};
        save_for_backward(out->ctx, saved_tensors, 2, NULL, 0);
        out->grad_fn = mul_backward;
    } 
    return out;
}

void mul_backward(Tensor* grad_output, Tensor* out) {
    Tensor* a = out->ctx->saved_tensors[0];
    Tensor* b = out->ctx->saved_tensors[1];

    if (a->requires_grad) {
        // grad_a = b * grad_output
        Matrix* grad_a = MatrixMultiply(grad_output->data, b->data);
        tensor_backward(a, grad_a);
        FreeMatrix(grad_a);
    }
    if (b->requires_grad) {
        // grad_b = a * grad_output
        Matrix* grad_b = MatrixMultiply(grad_output->data, a->data);
        tensor_backward(b, grad_b);
        FreeMatrix(grad_b);
    }
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_sum(Tensor* a) {
    // Sum of all elements of a Tensor
    Matrix* sum_matrix = InitMatrix(1, 1);
    sum_matrix->data[0] = MatrixSum(a->data); // MatrixSum return a double 

    Tensor* output = init_tensor(sum_matrix, a->requires_grad);

    if (output->requires_grad) {
        output->grad_fn = sum_backward;
        output->ctx = init_context(); 
        Tensor* saved_tensors[1] = {a};
        save_for_backward(output->ctx, saved_tensors, 1, NULL, 0);
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
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_negate(Tensor* input) {
    // Return a negative picture of all the elements of a Tensor
    Tensor* out = init_tensor(MatrixNeg(input->data), input->requires_grad);

    if (out->requires_grad) {
        out->ctx = init_context(); 
        out->grad_fn = neg_backward;
        Tensor* saved_tensors[1] = {input};
        save_for_backward(out->ctx, saved_tensors, 1, NULL, 0);
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
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_divide(Tensor* a, Tensor* b) {
    // out = a / b (element-wise)
    Tensor* out = init_tensor(MatrixDivide(a->data, b->data), a->requires_grad || b->requires_grad);

    if (out->requires_grad) {
        out->ctx = init_context();
        out->grad_fn = div_backward;
        Tensor* saved_tensors[2] = {a, b};
        save_for_backward(out->ctx, saved_tensors, 2, NULL, 0);
    }
    return out;
}

void div_backward(Tensor* grad_output, Tensor* out) {
    Tensor* a = out->ctx->saved_tensors[0];
    Tensor* b = out->ctx->saved_tensors[1];

    if (a->requires_grad) {
        // grad_a = grad_output / b 
        Matrix* grad_a = MatrixDivide(grad_output->data, b->data);
        tensor_backward(a, grad_a);
        FreeMatrix(grad_a);
    }
    if (b->requires_grad) {
        // grad_b = (grad_output * a) / b**2
        Matrix* b_sq = MatrixMultiply(b->data, b->data);
        Matrix* temp = MatrixMultiply(grad_output->data, a->data);
        Matrix* grad_b = MatrixNeg(MatrixDivide(temp, b_sq));

        tensor_backward(b, grad_b);

        FreeMatrix(b_sq);
        FreeMatrix(temp);
        FreeMatrix(grad_b);
    }
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    // out = a @ b (matrix multiplication)
    Tensor* out = init_tensor(MatrixMul(a->data, b->data), a->requires_grad || b->requires_grad);

    if (out->requires_grad) {
        out->ctx = init_context();
        out->grad_fn = matmul_backward;
        Tensor* saved_tensors[2] = {a, b};
        save_for_backward(out->ctx, saved_tensors, 2, NULL, 0);
    }
    return out;
}

void matmul_backward(Tensor* grad_output, Tensor* out) {
    Tensor* a = out->ctx->saved_tensors[0];
    Tensor* b = out->ctx->saved_tensors[1];

    if (a->requires_grad) {
        // grad_a = grad_output @ b.T 
        Matrix* grad_a = MatrixMul(grad_output->data, MatrixTranspose(b->data));
        tensor_backward(a, grad_a);
        FreeMatrix(grad_a);
    }
    if (b->requires_grad) {
        // grad_b = a.T @ grad_output 
        Matrix* grad_b = MatrixMul(MatrixTranspose(a->data), grad_output->data);
        tensor_backward(b, grad_b);
        FreeMatrix(grad_b);
    }
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_max(Tensor* input, int dim) {
    Tensor* output;
    if (dim == -1) {
        // Max over all the elements of a tensor
        Matrix* out = InitMatrix(1, 1);
        out->data[0] = MatrixMax(input->data);
        output = init_tensor(out, input->requires_grad); 
    } else {
        // Max along the specific dimension of a tensor
        output = init_tensor(MatrixMaxVals(input->data, dim), input->requires_grad);
    }
    
    if (output->requires_grad) {
            output->ctx = init_context();
            output->grad_fn = max_backward;
            Tensor* saved_tensors[1] = {input};
            save_for_backward(output->ctx, saved_tensors, 1, NULL, 0);
    }
    return output;
}

void max_backward(Tensor* grad_output, Tensor* out) {
    Tensor* input = out->ctx->saved_tensors[0];

    if (input->requires_grad) {
        Matrix* grad_input = ZerosMatrix(input->data->rows, input->data->cols);

        if (out->data->rows == 1 && out->data->cols == 1) {
            // Max over all the elements of a tensor
            double max_val = out->data->data[0];
            for (int i = 0; i < input->data->rows * input->data->cols; i++) {
                if (input->data->data[i] == max_val) {
                    grad_input->data[i] = grad_output->data->data[0];
                }
            }
        } else {
            // Max along a dimension of a tensor
            int dim = (out->data->rows == input->data->rows) ? 1 : 0;
            int other_dim = 1 - dim;
            int other_dim_size = (dim == 0) ? input->data->cols : input->data->rows;

            for (int i = 0; i < other_dim_size; i++) {
                double max_val = out->data->data[i];
                for (int j = 0; j < (dim == 0 ? input->data->rows : input->data->cols); j++) {
                    int index = (dim == 0) ? j * input->data->cols + i : i * input->data->cols + j;
                    if (input->data->data[index] == max_val) {
                        grad_input->data[index] = grad_output->data->data[i];
                    }
                }
            }
        }
        tensor_backward(input, grad_input);
        FreeMatrix(grad_input);
    }
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_min(Tensor* input, int dim) {
    Tensor* output;
    if (dim == -1) {
        // Max over all the elements of a tensor
        Matrix* out = InitMatrix(1, 1);
        out->data[0] = MatrixMin(input->data);
        output = init_tensor(out, input->requires_grad); 
    } else {
        // Max along the specific dimension of a tensor
        output = init_tensor(MatrixMinVals(input->data, dim), input->requires_grad);
    }
    
    if (output->requires_grad) {
            output->ctx = init_context();
            output->grad_fn = min_backward;
            Tensor* saved_tensors[1] = {input};
            save_for_backward(output->ctx, saved_tensors, 1, NULL, 0);
    }
    return output;
}

void min_backward(Tensor* grad_output, Tensor* out) {
    Tensor* input = out->ctx->saved_tensors[0];

    if (input->requires_grad) {
        Matrix* grad_input = ZerosMatrix(input->data->rows, input->data->cols);

        if (out->data->rows == 1 && out->data->cols == 1) {
            // Max over all the elements of a tensor
            double min_val = out->data->data[0];
            for (int i = 0; i < input->data->rows * input->data->cols; i++) {
                if (input->data->data[i] == min_val) {
                    grad_input->data[i] = grad_output->data->data[0];
                }
            }
        } else {
            // Max along a dimension of a tensor
            int dim = (out->data->rows == input->data->rows) ? 1 : 0;
            int other_dim = 1 - dim;
            int other_dim_size = (dim == 0) ? input->data->cols : input->data->rows;

            for (int i = 0; i < other_dim_size; i++) {
                double min_val = out->data->data[i];
                for (int j = 0; j < (dim == 0 ? input->data->rows : input->data->cols); j++) {
                    int index = (dim == 0) ? j * input->data->cols + i : i * input->data->cols + j;
                    if (input->data->data[index] == min_val) {
                        grad_input->data[index] = grad_output->data->data[i];
                    }
                }
            }
        }
        tensor_backward(input, grad_input);
        FreeMatrix(grad_input);
    }
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_pow(Tensor* input, double power) {
    // out = input**power;
    Tensor* out = init_tensor(MatrixPower(input->data, power), input->requires_grad);

    if (out->requires_grad) {
        out->ctx = init_context();
        out->grad_fn = pow_backward;
        Tensor* saved_tensors[1] = {input};
        double saved_scalars[1] = {power};
        save_for_backward(out->ctx, saved_tensors, 1, saved_scalars, 1);

        out->ctx->saved_scalars[0] = power;
    }
    return out;
}

void pow_backward(Tensor* grad_output, Tensor* out) {
    Tensor* input = out->ctx->saved_tensors[0];
    double power = out->ctx->saved_scalars[0];

    if (input->requires_grad) {
        // grad_input = power * (input)**(power-1) * grad_output
        Matrix* input_power = MatrixPower(input->data, power - 1);
        Matrix* times_power = MatrixScalarMul(input_power, power);
        Matrix* grad_input = MatrixMultiply(times_power, grad_output->data);

        tensor_backward(input, grad_input);

        FreeMatrix(grad_input);
        FreeMatrix(times_power);
        FreeMatrix(input_power);
    }
    
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_sqrt(Tensor* input) {
    return tensor_pow(input, 0.5);
}

Tensor* tensor_sin(Tensor* input) {
    // out = sin(input);
    Tensor* out = init_tensor(MatrixSin(input->data), input->requires_grad);

    if (out->requires_grad) {
        out->ctx = init_context();
        out->grad_fn = sin_backward;
        Tensor* saved_tensors[1] = {input};
        save_for_backward(out->ctx, saved_tensors, 1, NULL, 0);
    }
    return out;
}

void sin_backward(Tensor* grad_output, Tensor* out) {
    Tensor* input = out->ctx->saved_tensors[0];

    if (input->requires_grad) {
        // grad_input = cos(input) * grad_output 
        Matrix* grad_input = MatrixMultiply(MatrixCos(input->data), grad_output->data);
        tensor_backward(input, grad_input);
        FreeMatrix(grad_input);
    }
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_cos(Tensor* input) {
    // out = cos(input)
    Tensor* out = init_tensor(MatrixCos(input->data), input->requires_grad);

    if (out->requires_grad) {
        out->ctx = init_context();
        out->grad_fn = cos_backward;
        Tensor* saved_tensors[1] = {input};
        save_for_backward(out->ctx, saved_tensors, 1, NULL, 0);
    }
    return out;
}

void cos_backward(Tensor* grad_output, Tensor* out) {
    Tensor* input = out->ctx->saved_tensors[0];

    if (input->requires_grad) {
        // grad_input = -sin(input) * grad_output 
        Matrix* grad_input = MatrixMultiply(MatrixNeg(MatrixSin(input->data)), grad_output->data);
        tensor_backward(input, grad_input);
        FreeMatrix(grad_input);
    }
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_exp(Tensor* input) {
    // out = exp(input)
    Tensor* out = init_tensor(MatrixExp(input->data), input->requires_grad);

    if (out->requires_grad) {
        out->ctx = init_context();
        out->grad_fn = exp_backward;
        Tensor* saved_tensors[1] = {input};
        save_for_backward(out->ctx, saved_tensors, 1, NULL, 0);
    }
    return out;
}

void exp_backward(Tensor* grad_output, Tensor* out) {
    Tensor* input = out->ctx->saved_tensors[0];

    if (input->requires_grad) {
        // grad_input = exp(input) * grad_output 
        Matrix* grad_input = MatrixMultiply(MatrixExp(input->data), grad_output->data);
        tensor_backward(input, grad_input);
        FreeMatrix(grad_input);
    }
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_mean(Tensor* input, int dim) {
    Tensor* out;
    if (dim == -1) {
        // Mean over all the elements of a Tensor
        Matrix* mean_matrix = InitMatrix(1, 1);
        mean_matrix->data[0] = MatrixMean(input->data);
        out = init_tensor(mean_matrix, input->requires_grad);
    } else {
        // Mean along a specific dimension of a Tensor
        out = init_tensor(MatrixMeanVals(input->data, dim), input->requires_grad);
    }

    if (out->requires_grad) {
        out->ctx = init_context();
        out->grad_fn = mean_backward;
        Tensor* saved_tensors[1] = {input};
        save_for_backward(out->ctx, saved_tensors, 1, NULL, 0);
    }
    return out;
}

void mean_backward(Tensor* grad_output, Tensor* out) {
    Tensor* input = out->ctx->saved_tensors[0];

    if (input->requires_grad) {
        Matrix* grad_input = InitMatrix(input->data->rows, input->data->cols);
        int total_elems = MatrixNumel(input->data);

        if (out->data->rows == 1 && out->data->cols == 1) {
            double inv_total = 1.0 / total_elems;
            for (int i = 0; i < total_elems; i++) {
                grad_input->data[i] = inv_total * grad_output->data->data[0];
            }
        } else {
            int dim = (out->data->rows == input->data->rows) ? 1 : 0;
            int mean_dim_size = (dim == 0) ? input->data->rows : input->data->cols;
            double inv_mean_dim = 1.0 / mean_dim_size;
            
            for (int i = 0; i < total_elems; i++) {
                int index = (dim == 0) ? i % input->data->cols : i / input->data->cols;
                grad_input->data[i] = inv_mean_dim * grad_output->data->data[index];
            }
        }
        tensor_backward(input, grad_input);
        FreeMatrix(grad_input);
    }
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_std(Tensor* input, int dim) {
    Tensor* out;
    if (dim == -1) {
        // Mean over all the elements of a Tensor
        Matrix* std_matrix = InitMatrix(1, 1);
        std_matrix->data[0] = MatrixStd(input->data);
        out = init_tensor(std_matrix, input->requires_grad);
    } else {
        // Mean along a specific dimension of a Tensor
        out = init_tensor(MatrixStdVals(input->data, dim), input->requires_grad);
    }

    if (out->requires_grad) {
        out->ctx = init_context();
        out->grad_fn = std_backward;
        Tensor* saved_tensors[1] = {input};
        save_for_backward(out->ctx, saved_tensors, 1, NULL, 0);
    }
    return out;
}

void std_backward(Tensor* grad_output, Tensor* out) {
    Tensor* input = out->ctx->saved_tensors[0];

    if (input->requires_grad) {
        Matrix* grad_input = InitMatrix(input->data->rows, input->data->cols);
        int total_elems = MatrixNumel(input->data);

        if (out->data->rows == 1 && out->data->cols == 1) {
            // Standard deviation over all elements
            double mean = MatrixMean(input->data);
            double std = out->data->data[0];
            double inv_std = 1.0 / std;
            double inv_total = 1.0 / total_elems;

            for (int i = 0; i < total_elems; i++) {
                grad_input->data[i] = inv_total * inv_std * (input->data->data[i] - mean) * grad_output->data->data[0];
            }
        } else {
            // Standard deviation along a specific dimension
            int dim = (out->data->rows == input->data->rows) ? 1 : 0;
            int mean_dim_size = (dim == 0) ? input->data->rows : input->data->cols;
            double inv_mean_dim = 1.0 / mean_dim_size;

            // Compute the mean along the specified dimension
            Matrix* mean_vals = MatrixMeanVals(input->data, dim);

            for (int i = 0; i < total_elems; i++) {
                int index = (dim == 0) ? i % input->data->cols : i / input->data->cols;
                double std = out->data->data[index];
                double inv_std = 1.0 / std;

                grad_input->data[i] = inv_mean_dim * inv_std * (input->data->data[i] - mean_vals->data[index]) * grad_output->data->data[index];
            }

            FreeMatrix(mean_vals);
        }
        tensor_backward(input, grad_input);
        FreeMatrix(grad_input);
    }
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_var(Tensor* input, int dim) {
    return tensor_pow(tensor_std(input, dim), 2);
}

Tensor* tensor_transpose(Tensor* input) {
    Tensor* out = init_tensor(MatrixTranspose(input->data), input->requires_grad);

    if (out->requires_grad) {
        out->ctx = init_context();
        out->grad_fn = transpose_backward;
        Tensor* saved_tensors[1] = {input};
        save_for_backward(out->ctx, saved_tensors, 1, NULL, 0);
    }
    return out;
}

void transpose_backward(Tensor* grad_output, Tensor* out) {
    Tensor* input = out->ctx->saved_tensors[0];

    if (input->requires_grad) {
        Matrix* grad_input = MatrixTranspose(grad_output->data);
        tensor_backward(input, grad_input);
        FreeMatrix(grad_input);
    }
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_reshape(Tensor* input, int rows, int cols) {
    Tensor* out = init_tensor(MatrixReshape(input->data, rows, cols), input->requires_grad);

    if (out->requires_grad) {
        out->ctx = init_context();
        out->grad_fn = reshape_backward;
        Tensor* saved_tensors[1] = {input};
        save_for_backward(out->ctx, saved_tensors, 1, NULL, 0);
    }
    return out;
}

void reshape_backward(Tensor* grad_output, Tensor* out) {
    Tensor* input = out->ctx->saved_tensors[0];
    int new_rows = input->data->rows;
    int new_cols = input->data->cols;

    if (input->requires_grad) {
        Matrix* grad_input = MatrixReshape(grad_output->data, new_rows, new_cols);
        tensor_backward(input, grad_input);
        FreeMatrix(grad_input);
    }
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_concat(Tensor* a, Tensor* b, int dim) {
    Tensor* out = init_tensor(MatrixConcat(a->data, b->data, dim), a->requires_grad || b->requires_grad);

    if (out->requires_grad) {
        out->ctx = init_context();
        out->grad_fn = concat_backward;
        Tensor* saved_tensors[2] = {a, b};
        double saved_scalars[1] = {dim};
        save_for_backward(out->ctx, saved_tensors, 2, saved_scalars, 1);
    }
    return out;
}

void concat_backward(Tensor* grad_output, Tensor* out) {
    Tensor* a = out->ctx->saved_tensors[0];
    Tensor* b = out->ctx->saved_tensors[1];
    int dim = out->ctx->saved_scalars[0];

    if (a->requires_grad) {
        Matrix* grad_a = InitMatrix(a->data->rows, a->data->cols);
        if (dim == 0) { 
            // split along rows
            for (int i = 0; i < a->data->rows; i++) {
                for (int j = 0; j < a->data->cols; j++) {
                    MAT_AT(grad_a, i, j) = MAT_AT(grad_output->data, i, j);
                }
            }
        } else if (dim == 1) {
            // split along columns
            for (int i = 0; i < a->data->rows; i++) {
                for (int j = 0; j < a->data->cols; j++) {
                    MAT_AT(grad_a, i, j) = MAT_AT(grad_output->data, i, j);
                }
            }
        }
        tensor_backward(a, grad_a);
        FreeMatrix(grad_a);
    }
    if (b->requires_grad) {
        Matrix* grad_b = InitMatrix(b->data->rows, b->data->cols);
        if (dim == 0) {
            // split along rows
            for (int i = 0; i < b->data->rows; i++) {
                for (int j = 0; j < b->data->cols; j++) {
                    MAT_AT(grad_b, i, j) = MAT_AT(grad_output->data, (i + a->data->rows), j);
                }
            }
        } else if (dim == 1) {
            // split along columns
            for (int i = 0; i < b->data->rows; i++) {
                for (int j = 0; j < b->data->cols; j++) {
                    MAT_AT(grad_b, i, j) = MAT_AT(grad_output->data, i, (j + a->data->cols));
                }
            }
        }
        tensor_backward(b, grad_b);
        FreeMatrix(grad_b);
    }
    free_context(out->ctx);
    out->ctx = NULL;
}

Tensor* tensor_relu(Tensor* input) {
    // relu(in) = max(0, in)
    int rows = input->data->rows;
    int cols = input->data->cols;
    Tensor* out = init_tensor(InitMatrix(rows, cols), input->requires_grad);
    
    for (int i = 0; i < rows * cols; i++) {
        out->data->data[i] = fmax(input->data->data[i], 0); 
    }

    if (out->requires_grad) {
        out->ctx = init_context();
        out->grad_fn = relu_backward;
        Tensor* saved_tensors[1] = {input};
        save_for_backward(out->ctx, saved_tensors, 1, NULL, 0);
    }

    return out;
}

void relu_backward(Tensor* grad_output, Tensor* out) {
    Tensor* input = out->ctx->saved_tensors[0];

    if (input->requires_grad) {
        int size = input->data->rows * input->data->cols;
        Matrix* input_grad = InitMatrix(input->data->rows, input->data->cols);

        for (int i = 0; i < size; i++) {
            input_grad->data[i] = (input->data->data[i] > 0) ? grad_output->data->data[i] : 0;  
        }

        tensor_backward(input, input_grad);
        FreeMatrix(input_grad);
    }

    free_context(out->ctx);
    out->ctx = NULL;
}
