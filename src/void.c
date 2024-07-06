#include "void.h"

Tensor* init_tensor(Matrix* data, bool requires_grad) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    tensor->data = MatrixCopy(data);
    tensor->requires_grad = requires_grad;
    tensor->grad = NULL;
    tensor->_ctx = NULL;

    return tensor;
}

void free_tensor(Tensor* tensor) {
    if (tensor != NULL) return;
    if (tensor->data != NULL) FreeMatrix(tensor->data);
    if (tensor->grad != NULL) free_tensor(tensor->grad);
    if (tensor->_ctx != NULL) {
        if (tensor->_ctx->saved_tensors != NULL) 
            free(tensor->_ctx->saved_tensors);
        free(tensor->_ctx);
    }
    free(tensor);
}

Context* init_context(void(*_backward)(Context*, Tensor*), Tensor** saved_tensors, int tensor_count) {
    Context* ctx = (Context*)malloc(sizeof(Context));
    if (ctx == NULL) return NULL;

    ctx->_backward = _backward;
    ctx->saved_tensors = (Tensor**)malloc(tensor_count * sizeof(Tensor*));
    if (ctx->saved_tensors == NULL) {
        free(ctx);
        return NULL;
    }

    memcpy(ctx->saved_tensors, saved_tensors, tensor_count * sizeof(Tensor*));
    ctx->saved_tensors_count = tensor_count;

    return ctx;
}

void build_topo(Tensor* tensor, Tensor** visited, int* visited_count, Tensor** nodes, int* nodes_count) {
    bool found = false;
    for (int i = 0; i < *visited_count; i++) {
        if (visited[i] == tensor) {
            found = true;
            break;
        }
    }
    if (!found) {
        visited[*visited_count] = tensor;
        (*visited_count)++;
        if (tensor->_ctx) {
            for (int i = 0; i < tensor->_ctx->saved_tensors_count; i++) {
                build_topo(tensor->_ctx->saved_tensors[i], visited, visited_count, nodes, nodes_count);
            }
        }
        nodes[*nodes_count] = tensor;
        (*nodes_count)++;
    }
}

void backward(Tensor* tensor) {
    if (!tensor->_ctx) return;
    if (tensor->data->rows * tensor->data->cols != 1) {
        fprintf(stderr, "backward can only be called for scalar tensors, but got a tensor of shape (%d, %d)\n", tensor->data->rows, tensor->data->cols);
        exit(1);
    }
    if (!tensor->grad) tensor->grad = tensor_ones(1, 1, false);

    Tensor* visited[1000]; // assuming max of 1000 tensors 
    int visited_count = 0;
    Tensor* nodes[1000];
    int nodes_count = 0;
    build_topo(tensor, visited, &visited_count, nodes, &nodes_count);

    for (int i = nodes_count - 1; i >= 0; i--) {
        Tensor* t = nodes[i];
        if (t->_ctx) {
            t->_ctx->_backward(t->_ctx, t->grad);
        }
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

    for (int i = 0; i < t->grad->data->rows; i++) {
        for (int j = 0; j < t->grad->data->cols; j++) {
            double val = fabs(MAT_AT(t->grad->data, i, j));
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
        for (int i = 0; i < t->grad->data->rows; i++) {
            if (i == 0) {
                printf("[");
            } else {
                printf(" [");
            }
            for (int j = 0; j < t->grad->data->cols; j++) {
                printf("%*.*f ", max_digits, 5, MAT_AT(t->grad->data, i, j));
            }
            if (i != t->grad->data->rows-1) {
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
    Tensor* out = init_tensor(
            MatrixAdd(a->data, b->data), a->requires_grad || b->requires_grad
    );

    if (out->requires_grad) {
        Tensor* saved_tensors[2] = {a, b};
        out->_ctx = init_context(add_backward, saved_tensors, 2);
    }
    return out;
}

void add_backward(Context* ctx, Tensor* grad_output) {
    Tensor* a = ctx->saved_tensors[0];
    Tensor* b = ctx->saved_tensors[1];

    if (a->requires_grad) {
        a->grad = init_tensor(MatrixZerosLike(a->data), false);
        a->grad->data = grad_output->data;
    } 
    if (b->requires_grad) {
        b->grad = init_tensor(MatrixZerosLike(b->data), false);
        b->grad->data = grad_output->data;
    }
}

Tensor* tensor_sub(Tensor* a, Tensor* b) {
    // out = a + b
    Tensor* out = init_tensor(
            MatrixSub(a->data, b->data), a->requires_grad || b->requires_grad
    );

    if (out->requires_grad) {
        Tensor* saved_tensors[2] = {a, b};
        out->_ctx = init_context(sub_backward, saved_tensors, 2);
    }
    return out;
}

void sub_backward(Context* ctx, Tensor* grad_output) {
    Tensor* a = ctx->saved_tensors[0];
    Tensor* b = ctx->saved_tensors[1];

    if (a->requires_grad) {
        a->grad = init_tensor(MatrixZerosLike(a->data), false);
        a->grad->data = grad_output->data;
    } 
    if (b->requires_grad) {
        b->grad = init_tensor(MatrixZerosLike(b->data), false);
        b->grad->data = MatrixNeg(grad_output->data);
    }
}

Tensor* tensor_multiply(Tensor* a, Tensor* b) {
    // out = a*b
    Tensor* out = init_tensor(
        MatrixMultiply(a->data, b->data), a->requires_grad || b->requires_grad
    );

    if (out->requires_grad) {
        Tensor* saved_tensors[2] = {a, b};
        out->_ctx = init_context(mul_backward, saved_tensors, 2);
    } 
    return out;
}

void mul_backward(Context* ctx, Tensor* grad_output) {
    Tensor* a = ctx->saved_tensors[0];
    Tensor* b = ctx->saved_tensors[1];

    if (a->requires_grad) {
        // grad_a = b * grad_output
        a->grad = init_tensor(MatrixZerosLike(a->data), false);
        a->grad->data = MatrixMultiply(grad_output->data, b->data);
    }
    if (b->requires_grad) {
        // grad_b = a * grad_output
        b->grad = init_tensor(MatrixZerosLike(b->data), false);
        b->grad->data = MatrixMultiply(grad_output->data, a->data);
    }
}

Tensor* tensor_sum(Tensor* a) {
    // Sum of all elements of a Tensor
    Matrix* sum_matrix = InitMatrix(1, 1);
    sum_matrix->data[0] = MatrixSum(a->data); // MatrixSum returns a double 

    Tensor* output = init_tensor(sum_matrix, a->requires_grad);

    if (output->requires_grad) {
        Tensor* saved_tensors[1] = {a};
        output->_ctx = init_context(sum_backward, saved_tensors, 1);
    }
    return output;
}

void sum_backward(Context* ctx, Tensor* grad_output) {
    Tensor* input = ctx->saved_tensors[0];

    if (input->requires_grad) {
        input->grad = init_tensor(MatrixOnesLike(input->data), false);
        input->grad->data = MatrixScalarMul(input->grad->data, grad_output->data->data[0]);
    }
}

Tensor* tensor_negate(Tensor* input) {
    // Return a negative picture of all the elements of a Tensor
    Tensor* out = init_tensor(MatrixNeg(input->data), input->requires_grad);

    if (out->requires_grad) {
        Tensor* saved_tensors[1] = {input};
        out->_ctx = init_context(neg_backward, saved_tensors, 1);
    }
    return out;
}

void neg_backward(Context* ctx, Tensor* grad_output) {
    Tensor* input = ctx->saved_tensors[0];

    if (input->requires_grad) {
        input->grad = init_tensor(MatrixOnesLike(input->data), false);
        input->grad->data = MatrixScalarMul(input->grad->data, grad_output->data->data[0]);
    }
}

Tensor* tensor_divide(Tensor* a, Tensor* b) {
    // out = a / b (element-wise)
    Tensor* out = init_tensor(MatrixDivide(a->data, b->data), a->requires_grad || b->requires_grad);

    if (out->requires_grad) {
        Tensor* saved_tensors[2] = {a, b};
        out->_ctx = init_context(div_backward, saved_tensors, 2);
    }
    return out;
}

void div_backward(Context* ctx, Tensor* grad_output) {
    Tensor* a = ctx->saved_tensors[0];
    Tensor* b = ctx->saved_tensors[1];

    if (a->requires_grad) {
        // grad_a = grad_output / b 
        a->grad = init_tensor(MatrixDivide(grad_output->data, b->data), false);
    }
    if (b->requires_grad) {
        // grad_b = (grad_output * a) / b**2
        Matrix* b_sq = MatrixMultiply(b->data, b->data);
        Matrix* temp = MatrixMultiply(grad_output->data, a->data);
        b->grad = init_tensor(MatrixNeg(MatrixDivide(temp, b_sq)), false);

        FreeMatrix(b_sq);
        FreeMatrix(temp);
    }
}

Tensor* tensor_matmul(Tensor* a, Tensor* b) {
    // out = a @ b (matrix multiplication)
    Tensor* out = init_tensor(MatrixMul(a->data, b->data), a->requires_grad || b->requires_grad);

    if (out->requires_grad) {
        Tensor* saved_tensors[2] = {a, b};
        out->_ctx = init_context(matmul_backward, saved_tensors, 2);
    }
    return out;
}

void matmul_backward(Context* ctx, Tensor* grad_output) {
    Tensor* a = ctx->saved_tensors[0];
    Tensor* b = ctx->saved_tensors[1];

    if (a->requires_grad) {
        // grad_a = grad_output @ b.T 
        a->grad = init_tensor(
                MatrixMul(grad_output->data, MatrixTranspose(b->data)), 
                true
        );
    }
    if (b->requires_grad) {
        // grad_b = a.T @ grad_output 
        b->grad = init_tensor(
                MatrixMul(MatrixTranspose(a->data), grad_output->data),
                true
        );
    }
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
            Tensor* saved_tensors[1] = {input};
            output->_ctx = init_context(max_backward, saved_tensors, 1);
    }
    return output;
}

void max_backward(Context* ctx, Tensor* grad_output) {
    Tensor* input = ctx->saved_tensors[0];

    if (input->requires_grad) {
        input->grad = init_tensor(
                ZerosMatrix(input->data->rows, input->data->cols),
                true
        );

        if (grad_output->data->rows == 1 && grad_output->data->cols == 1) {
            // Max over all the elements of a tensor
            double max_val = grad_output->data->data[0];
            for (int i = 0; i < input->data->rows * input->data->cols; i++) {
                if (input->data->data[i] == max_val) {
                    input->grad->data->data[i] = grad_output->data->data[0];
                }
            }
        } else {
            // Max along a dimension of a tensor
            int dim = (grad_output->data->rows == input->data->rows) ? 1 : 0;
            int other_dim = 1 - dim;
            int other_dim_size = (dim == 0) ? input->data->cols : input->data->rows;

            for (int i = 0; i < other_dim_size; i++) {
                double max_val = grad_output->data->data[i];
                for (int j = 0; j < (dim == 0 ? input->data->rows : input->data->cols); j++) {
                    int index = (dim == 0) ? j * input->data->cols + i : i * input->data->cols + j;
                    if (input->data->data[index] == max_val) {
                        input->grad->data->data[index] = grad_output->data->data[i];
                    }
                }
            }
        }
    }
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
            Tensor* saved_tensors[1] = {input};
            output->_ctx = init_context(min_backward, saved_tensors, 1);
    }
    return output;
}

void min_backward(Context* ctx, Tensor* grad_output) {
    Tensor* input = ctx->saved_tensors[0];

    if (input->requires_grad) {
        input->grad = init_tensor(ZerosMatrix(input->data->rows, input->data->cols), false);

        if (grad_output->data->rows == 1 && grad_output->data->cols == 1) {
            // Max over all the elements of a tensor
            double min_val = grad_output->data->data[0];
            for (int i = 0; i < input->data->rows * input->data->cols; i++) {
                if (input->data->data[i] == min_val) {
                    input->grad->data->data[i] = grad_output->data->data[0];
                }
            }
        } else {
            // Max along a dimension of a tensor
            int dim = (grad_output->data->rows == input->data->rows) ? 1 : 0;
            int other_dim = 1 - dim;
            int other_dim_size = (dim == 0) ? input->data->cols : input->data->rows;

            for (int i = 0; i < other_dim_size; i++) {
                double min_val = grad_output->data->data[i];
                for (int j = 0; j < (dim == 0 ? input->data->rows : input->data->cols); j++) {
                    int index = (dim == 0) ? j * input->data->cols + i : i * input->data->cols + j;
                    if (input->data->data[index] == min_val) {
                        input->grad->data->data[index] = grad_output->data->data[i];
                    }
                }
            }
        }
    }
}

Tensor* tensor_pow(Tensor* input, double power) {
    // out = input**power;
    Tensor* out = init_tensor(MatrixPower(input->data, power), input->requires_grad);

    if (out->requires_grad) {
        Tensor* saved_tensors[2] = {input, out};
        out->_ctx = init_context(pow_backward, saved_tensors, 2);
    }
    return out;
}

void pow_backward(Context* ctx, Tensor* grad_output) {
    Tensor* input = ctx->saved_tensors[0];
    Tensor* output = ctx->saved_tensors[1];
    double power = log(output->data->data[0]) / log(input->data->data[0]);

    if (input->requires_grad) {
        // grad_input = power * (input)**(power-1) * grad_output
        Matrix* input_power = MatrixPower(input->data, power - 1);
        Matrix* times_power = MatrixScalarMul(input_power, power);
        input->grad = init_tensor(MatrixMultiply(times_power, grad_output->data), false);

        FreeMatrix(times_power);
        FreeMatrix(input_power);
    }
}

Tensor* tensor_sqrt(Tensor* input) {
    return tensor_pow(input, 0.5);
}

Tensor* tensor_log(Tensor* input) {
    Tensor* out = init_tensor(MatrixLog(input->data), input->requires_grad);
    
    if (out->requires_grad) {
        Tensor* saved_tensors[1] = {input};
        out->_ctx = init_context(log_backward, saved_tensors, 1);
    }
    return out;
}

void log_backward(Context* ctx, Tensor* grad_output) {
    Tensor* input = ctx->saved_tensors[0];
    if (input->requires_grad) {
        // grad_input = 1 / input 
        input->grad = init_tensor(MatrixReciprocal(input->data), false);
    }
}

Tensor* tensor_sin(Tensor* input) {
    // out = sin(input);
    Tensor* out = init_tensor(MatrixSin(input->data), input->requires_grad);

    if (out->requires_grad) {
        Tensor* saved_tensors[1] = {input};
        out->_ctx = init_context(sin_backward, saved_tensors, 1);
    }
    return out;
}

void sin_backward(Context* ctx, Tensor* grad_output) {
    Tensor* input = ctx->saved_tensors[0];

    if (input->requires_grad) {
        // grad_input = cos(input) * grad_output 
        input->grad = init_tensor(MatrixMultiply(MatrixCos(input->data), grad_output->data), false);
    }
}

Tensor* tensor_cos(Tensor* input) {
    // out = cos(input)
    Tensor* out = init_tensor(MatrixCos(input->data), input->requires_grad);

    if (out->requires_grad) {
        Tensor* saved_tensors[1] = {input};
        out->_ctx = init_context(cos_backward, saved_tensors, 1);
    }
    return out;
}

void cos_backward(Context* ctx, Tensor* grad_output) {
    Tensor* input = ctx->saved_tensors[0];

    if (input->requires_grad) {
        // grad_input = -sin(input) * grad_output 
        input->grad = init_tensor(
                MatrixMultiply(MatrixNeg(MatrixSin(input->data)), grad_output->data),
                false
        );
    }
}

Tensor* tensor_exp(Tensor* input) {
    // out = exp(input)
    Tensor* out = init_tensor(MatrixExp(input->data), input->requires_grad);

    if (out->requires_grad) {
        Tensor* saved_tensors[1] = {input};
        out->_ctx = init_context(exp_backward, saved_tensors, 1);
    }
    return out;
}

void exp_backward(Context* ctx, Tensor* grad_output) {
    Tensor* input = ctx->saved_tensors[0];

    if (input->requires_grad) {
        // grad_input = exp(input) * grad_output 
        input->grad = init_tensor(MatrixMultiply(MatrixExp(input->data), grad_output->data), false);
    }
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
        Tensor* saved_tensors[2] = {input, out};
        out->_ctx = init_context(mean_backward, saved_tensors, 2);
    }
    return out;
}

void mean_backward(Context* ctx, Tensor* grad_output) {
    Tensor* input = ctx->saved_tensors[0];
    Tensor* out = ctx->saved_tensors[1];

    if (input->requires_grad) {
        input->grad = init_tensor(InitMatrix(input->data->rows, input->data->cols), false);
        int total_elems = MatrixNumel(input->data);

        if (out->data->rows == 1 && out->data->cols == 1) {
            double inv_total = 1.0 / total_elems;
            for (int i = 0; i < total_elems; i++) {
                input->grad->data->data[i] = inv_total * grad_output->data->data[0];
            }
        } else {
            int dim = (out->data->rows == input->data->rows) ? 1 : 0;
            int mean_dim_size = (dim == 0) ? input->data->rows : input->data->cols;
            double inv_mean_dim = 1.0 / mean_dim_size;
            
            for (int i = 0; i < total_elems; i++) {
                int index = (dim == 0) ? i % input->data->cols : i / input->data->cols;
                input->grad->data->data[i] = inv_mean_dim * grad_output->data->data[index];
            }
        }
    }
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
        Tensor* saved_tensors[2] = {input, out};
        out->_ctx = init_context(std_backward, saved_tensors, 2);
    }
    return out;
}

void std_backward(Context* ctx, Tensor* grad_output) {
    Tensor* input = ctx->saved_tensors[0];
    Tensor* out = ctx->saved_tensors[1];

    if (input->requires_grad) {
        input->grad = init_tensor(InitMatrix(input->data->rows, input->data->cols), false);
        int total_elems = MatrixNumel(input->data);

        if (out->data->rows == 1 && out->data->cols == 1) {
            // Standard deviation over all elements
            double mean = MatrixMean(input->data);
            double std = out->data->data[0];
            double inv_std = 1.0 / std;
            double inv_total = 1.0 / total_elems;

            for (int i = 0; i < total_elems; i++) {
                input->grad->data->data[i] = inv_total * inv_std * (input->data->data[i] - mean) * grad_output->data->data[0];
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

                input->grad->data->data[i] = inv_mean_dim * inv_std * (input->data->data[i] - mean_vals->data[index]) * grad_output->data->data[index];
            }

            FreeMatrix(mean_vals);
        }
    }
}

Tensor* tensor_var(Tensor* input, int dim) {
    return tensor_pow(tensor_std(input, dim), 2);
}

Tensor* tensor_transpose(Tensor* input) {
    Tensor* out = init_tensor(MatrixTranspose(input->data), input->requires_grad);

    if (out->requires_grad) {
        Tensor* saved_tensors[1] = {input};
        out->_ctx = init_context(transpose_backward, saved_tensors, 1);
    }
    return out;
}

void transpose_backward(Context* ctx, Tensor* grad_output) {
    Tensor* input = ctx->saved_tensors[0];

    if (input->requires_grad) {
        input->grad = init_tensor(MatrixTranspose(grad_output->data), false);
    }
}

Tensor* tensor_reshape(Tensor* input, int rows, int cols) {
    Tensor* out = init_tensor(MatrixReshape(input->data, rows, cols), input->requires_grad);

    if (out->requires_grad) {
        Tensor* saved_tensors[1] = {input};
        out->_ctx = init_context(reshape_backward, saved_tensors, 1);
    }
    return out;
}

void reshape_backward(Context* ctx, Tensor* grad_output) {
    Tensor* input = ctx->saved_tensors[0];
    int new_rows = input->data->rows;
    int new_cols = input->data->cols;

    if (input->requires_grad) {
        input->grad = init_tensor(MatrixReshape(grad_output->data, new_rows, new_cols), false);
    }
}

Tensor* tensor_concat(Tensor* a, Tensor* b, int dim) {
    Tensor* out = init_tensor(MatrixConcat(a->data, b->data, dim), a->requires_grad || b->requires_grad);

    if (out->requires_grad) {
        Tensor* saved_tensors[3] = {a, b, out};
        out->_ctx = init_context(concat_backward, saved_tensors, 3);
    }
    return out;
}

void concat_backward(Context* ctx, Tensor* grad_output) {
    Tensor* a = ctx->saved_tensors[0];
    Tensor* b = ctx->saved_tensors[1];
    Tensor* out = ctx->saved_tensors[2];
    int dim = (out->data->rows == a->data->rows) ? 0 : 1;

    if (a->requires_grad) {
        a->grad = init_tensor(InitMatrix(a->data->rows, a->data->cols), false);
        if (dim == 0) { 
            // split along rows
            for (int i = 0; i < a->data->rows; i++) {
                for (int j = 0; j < a->data->cols; j++) {
                    MAT_AT(a->grad->data, i, j) = MAT_AT(grad_output->data, i, j);
                }
            }
        } else if (dim == 1) {
            // split along columns
            for (int i = 0; i < a->data->rows; i++) {
                for (int j = 0; j < a->data->cols; j++) {
                    MAT_AT(a->grad->data, i, j) = MAT_AT(grad_output->data, i, j);
                }
            }
        }
    }
    if (b->requires_grad) {
        b->grad = init_tensor(InitMatrix(b->data->rows, b->data->cols), false);
        if (dim == 0) {
            // split along rows
            for (int i = 0; i < b->data->rows; i++) {
                for (int j = 0; j < b->data->cols; j++) {
                    MAT_AT(b->grad->data, i, j) = MAT_AT(grad_output->data, (i + a->data->rows), j);
                }
            }
        } else if (dim == 1) {
            // split along columns
            for (int i = 0; i < b->data->rows; i++) {
                for (int j = 0; j < b->data->cols; j++) {
                    MAT_AT(b->grad->data, i, j) = MAT_AT(grad_output->data, i, (j + a->data->cols));
                }
            }
        }
    }
}

Tensor* tensor_slice(Tensor* tensor, int from_rows, int to_rows, int from_cols, int to_cols) {
    Tensor* out = init_tensor(
        MatrixSlice(tensor->data, from_rows, to_rows, from_cols, to_cols),
        tensor->requires_grad
    );
    return out;
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
        Tensor* saved_tensors[1] = {input};
        out->_ctx = init_context(relu_backward, saved_tensors, 1);
    }

    return out;
}

void relu_backward(Context* ctx, Tensor* grad_output) {
    Tensor* input = ctx->saved_tensors[0];

    if (input->requires_grad) {
        int size = input->data->rows * input->data->cols;
        input->grad = init_tensor(InitMatrix(input->data->rows, input->data->cols), false);

        for (int i = 0; i < size; i++) {
            input->grad->data->data[i] = (input->data->data[i] > 0) ? grad_output->data->data[i] : 0;  
        }
    }
}
