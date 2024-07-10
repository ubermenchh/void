#ifndef void_h
#define void_h

#include <flash.h>
#include <stdbool.h>

#define TABLE_SIZE 1024

typedef struct Tensor Tensor;
typedef struct Context Context;

struct Tensor {
    Matrix* data;
    bool requires_grad;
    
    Tensor* grad;
    Context* _ctx;
};

struct Context {
    void (*_backward) (Context*, Tensor* grad_output);
    Tensor** saved_tensors;
    int saved_tensors_count;
};



// Utility Functions
Tensor* init_tensor(Matrix* data, bool requires_grad);
void free_tensor(Tensor* t);
void print_tensor(Tensor* t);
void print_tensor_grad(Tensor* t);
void tensor_shape(Tensor* t);

// Backward function
Context* init_context(void(*_backward)(Context*, Tensor*), Tensor** saved_tensors, int tensor_count);
void backward(Tensor* tensor);

// Tensor Initialization
Tensor* tensor_rand(int rows, int cols, bool requires_grad, int seed);
Tensor* tensor_randn(int rows, int cols, bool requires_grad, int seed);
Tensor* tensor_ones(int rows, int cols, bool requires_grad);
Tensor* tensor_zeros(int rows, int cols, bool requires_grad);
Tensor* tensor_eye(int size, bool requires_grad);
Tensor* tensor_ones_like(Tensor* t, bool requires_grad);
Tensor* tensor_zeros_like(Tensor* t, bool requires_grad);
Tensor* tensor_full(int rows, int cols, double value, bool requires_grad);
Tensor* tensor_full_like(Tensor* t, double value, bool requires_grad);
Tensor* tensor_mask(int rows, int cols, double prob, bool requires_grad);

/* Tensor Operations */

// Adds two tensors, elementwise
Tensor* tensor_add(Tensor* a, Tensor* b);
void add_backward(Context* ctx, Tensor* grad_output);
// Elementwise difference of two tensors
Tensor* tensor_sub(Tensor* a, Tensor* b);
void sub_backward(Context* ctx, Tensor* grad_output);
// Multiplies two tensors, elementwise
Tensor* tensor_multiply(Tensor* a, Tensor* b);
void mul_backward(Context* ctx, Tensor* grad_output);
// Calculates the sum of the elements of a tensor
Tensor* tensor_sum(Tensor* a, int dim);
void sum_backward(Context* ctx, Tensor* grad_output);
// Negates all the elements of a tensor
Tensor* tensor_negate(Tensor* a);
void neg_backward(Context* ctx, Tensor* grad_output);
// Divides two tensors, elementwise
Tensor* tensor_divide(Tensor* a, Tensor* b);
void div_backward(Context* ctx, Tensor* grad_output);
// Calculates the Matrix Multiplication of 2 tensors
Tensor* tensor_matmul(Tensor* a, Tensor* b);
void matmul_backward(Context* ctx, Tensor* grad_output);
// Calculates the maximum of a tensor, (-1: all-elements, 0: row-wise, 1: col_wise)
Tensor* tensor_max(Tensor* input, int dim);
void max_backward(Context* ctx, Tensor* grad_output);
// Calculates the minimum of the tensor, (-1: all-elements, 0: row-wise, 1: col_wise)
Tensor* tensor_min(Tensor* input, int dim);
void min_backward(Context* ctx, Tensor* grad_output);
// Calculates x^y of the elements of a tensor
Tensor* tensor_pow(Tensor* input, double power);
void pow_backward(Context* ctx, Tensor* grad_output);
// Calculates the log(log10) of the elements of a tensor
Tensor* tensor_log(Tensor* input);
void log_backward(Context* ctx, Tensor* grad_output);
// Calculates the sqrt of elements of a tensor
Tensor* tensor_sqrt(Tensor* input);
// Calculates sine of the elements of a tensor
Tensor* tensor_sin(Tensor* input);
void sin_backward(Context* ctx, Tensor* grad_output);
// Calculates the cosine of the elements of a tensor
Tensor* tensor_cos(Tensor* input);
void cos_backward(Context* ctx, Tensor* grad_output);
// Calculates e^x of the elements of a tensor
Tensor* tensor_exp(Tensor* input);
void exp_backward(Context* ctx, Tensor* grad_output);
// Calculates mean of a tensor, (-1: all elements, 0: row-wise, 1: col_wise)
Tensor* tensor_mean(Tensor* input, int dim);
void mean_backward(Context* ctx, Tensor* grad_output);
// Calculate standard deviation of a tensor, (-1: all elements, 0: row-wise, 1: col-wise)
Tensor* tensor_std(Tensor* input, int dim);
void std_backward(Context* ctx, Tensor* grad_output);
// Calculates the variance of the tensor, (-1: all the elements, 0: row-wise, 1: col_wise)
Tensor* tensor_var(Tensor* input, int dim);
// Transposes the given tensor
Tensor* tensor_transpose(Tensor* input);
void transpose_backward(Context* ctx, Tensor* grad_output);
// Reshape a given tensor
Tensor* tensor_reshape(Tensor* input, int rows, int cols);
void reshape_backward(Context* ctx, Tensor* grad_output);
// Concatenates two tensors along a dim (0: row-wise, 1: col-wise)
Tensor* tensor_concat(Tensor* a, Tensor* b, int dim);
void concat_backward(Context* ctx, Tensor* grad_output);

Tensor* tensor_slice(Tensor* tensor, int from_rows, int to_rows, int from_cols, int to_cols);
Tensor* tensor_broadcast(Tensor* tensor, int rows, int cols);

Tensor* tensor_scale(Tensor* tensor, double scale);
void scale_backward(Context* ctx, Tensor* grad_output);

// ReLU Activation Function
Tensor* tensor_relu(Tensor* input);
void relu_backward(Context* ctx, Tensor* grad_output);
// Mean Squared Error
Tensor* mse(Tensor* y_pred, Tensor* y_true);
void mse_backward(Context* ctx, Tensor* grad_output);
// Cross-Entropy Loss 
Tensor* ce_loss(Tensor* output, Tensor* target);
void ce_backward(Context* ctx, Tensor* grad_output);
// Softmax
Tensor* softmax(Tensor* tensor);
void softmax_backward(Context* ctx, Tensor* grad_output);

Tensor* softmax_cross_entropy(Tensor* logits, Tensor* target);
void softmax_cross_entropy_backward(Context* ctx, Tensor* grad_output);

    /* Layers */
typedef struct Module Module;
typedef struct Linear Linear;
typedef struct Optim Optim;
typedef struct SGD SGD;

struct Module {
    void* impl;
    Tensor* (*forward)(Module* self, Tensor* input);
    Tensor** (*parameters)(Module* self, int* count);
    void (*free)(Module* self);
};
struct Linear {
    Module base;
    int in_dim;
    int out_dim;

    Tensor* weight;
    Tensor* bias;
    bool has_bias;
};
struct Optim {
    void* impl;
    void (*step)(Optim* self);
    void (*zero_grad)(Optim* self);
    void (*free)(Optim* self);
};
struct SGD {
    Optim base;
    Tensor** params;
    int param_count;
    double lr;
};

Module* init_linear(int in_dim, int out_dim, bool has_bias);
Tensor* linear_forward(Module* module, Tensor* input);
Tensor** linear_parameters(Module* module, int* count);
void free_linear(Module* module); 

Optim* init_sgd(Tensor** params, int param_count, double lr);
void sgd_step(Optim* optim);
void sgd_zero_grad(Optim* optim);
void free_sgd(Optim* optim);

#endif // void_h
