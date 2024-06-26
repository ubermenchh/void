#include <flash.h>
#include <stdbool.h>


typedef struct Tensor {
    Matrix* data;
    Matrix* grad;
    bool requires_grad;
    void (*grad_fn)(struct Tensor*, struct Tensor*);
    struct Context* ctx;
} Tensor;

typedef struct Context {
    Tensor** saved_tensors;
    int num_saved_tensors;
    double* saved_scalars; 
    int num_saved_scalars;
} Context;

// Utility Functions
Tensor* init_tensor(Matrix* data, bool requires_grad);
void free_tensor(Tensor* t);
void save_for_backward(Context* ctx, Tensor** tensors, int num_tensors, double* scalars, int num_scalars);
void print_tensor(Tensor* t);
void print_tensor_grad(Tensor* t);
void tensor_backward(Tensor* t, Matrix* grad);
void tensor_shape(Tensor* t);
Context* init_context();
void free_context(Context* ctx);

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

// Tensor Operations
Tensor* tensor_add(Tensor* a, Tensor* b);
void add_backward(Tensor* grad_out, Tensor* out);

Tensor* tensor_sub(Tensor* a, Tensor* b);

Tensor* tensor_multiply(Tensor* a, Tensor* b);
void mul_backward(Tensor* grad_out, Tensor* out);

Tensor* tensor_sum(Tensor* a);
void sum_backward(Tensor* grad_out, Tensor* out);

Tensor* tensor_negate(Tensor* a);
void neg_backward(Tensor* grad_out, Tensor* out);

Tensor* tensor_divide(Tensor* a, Tensor* b);
void div_backward(Tensor* grad_out, Tensor* out);

Tensor* tensor_matmul(Tensor* a, Tensor* b);
void matmul_backward(Tensor* grad_out, Tensor* out);

Tensor* tensor_max(Tensor* input, int dim);
void max_backward(Tensor* grad_output, Tensor* out);

Tensor* tensor_min(Tensor* input, int dim);
void min_backward(Tensor* grad_output, Tensor* out);

Tensor* tensor_pow(Tensor* input, double power);
void pow_backward(Tensor* grad_output, Tensor* out);

Tensor* tensor_log(Tensor* input);
void log_backward(Tensor* grad_output, Tensor* out);

Tensor* tensor_sqrt(Tensor* input);

Tensor* tensor_sin(Tensor* input);
void sin_backward(Tensor* grad_output, Tensor* out);

Tensor* tensor_cos(Tensor* input);
void cos_backward(Tensor* grad_output, Tensor* out);

Tensor* tensor_exp(Tensor* input);
void exp_backward(Tensor* grad_output, Tensor* out);

Tensor* tensor_mean(Tensor* input, int dim);
void mean_backward(Tensor* grad_output, Tensor* out);

Tensor* tensor_std(Tensor* input, int dim);
void std_backward(Tensor* grad_output, Tensor* out);

Tensor* tensor_var(Tensor* input, int dim);

Tensor* tensor_transpose(Tensor* input);
void transpose_backward(Tensor* grad_output, Tensor* out);

Tensor* tensor_reshape(Tensor* input, int rows, int cols);
void reshape_backward(Tensor* grad_output, Tensor* out);

Tensor* tensor_concat(Tensor* a, Tensor* b, int dim);
void concat_backward(Tensor* grad_output, Tensor* out);

Tensor* tensor_relu(Tensor* input);
void relu_backward(Tensor* grad_output, Tensor* out);

// Layers
typedef struct {
    int in_dim;
    int out_dim;
    bool has_bias;

    Tensor* weight;
    Tensor* bias;

    Tensor** parameters;
} Linear;

Linear* init_linear(int in_dim, int out_dim, bool has_bias);
void free_linear(Linear* lin);
Tensor* linear_forward(Linear* lin, Tensor* input);

typedef struct {
    int n_embed;
    Tensor* gamma;
    Tensor* beta;
} LayerNorm;

LayerNorm* init_layernorm(int n_embed);
void free_layernorm(LayerNorm* ln);
Tensor* layernorm_forward(LayerNorm* ln, Tensor* input);

typedef struct {
    double drop_prob;
    bool train_mode;
    Tensor* mask;
} Dropout;

Dropout* init_dropout(double drop_prob);
void free_dropout(Dropout* dp);
Tensor* dropout_forward(Dropout* dp, Tensor* input);

Tensor* mse(Tensor* y_true, Tensor* y_pred);
void mse_backward(Tensor* grad_output, Tensor* out);
