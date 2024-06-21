#include <flash.h>
#include <stdbool.h>

#define INIT_CONTEXT (Context*)malloc(sizeof(Context))

typedef struct Tensor {
    Matrix* data;
    Matrix* grad;
    bool requires_grad;
    void (*grad_fn)(struct Tensor*, struct Tensor*);
    struct Context* ctx;
} Tensor;

typedef struct Context {
    Tensor** saved_tensors;
    int num_saved;
} Context;

// Utility Functions
Tensor* init_tensor(Matrix* data, bool requires_grad);
void free_tensor(Tensor* t);
void save_for_backward(Context* ctx, Tensor** tensors, int num_saved);
void print_tensor(Tensor* t);
void print_tensor_grad(Tensor* t);
void tensor_backward(Tensor* t, Matrix* grad);

// Operations
Tensor* add(Tensor* a, Tensor* b);
void add_backward(Tensor* grad_out, Tensor* out);

Tensor* multiply(Tensor* a, Tensor* b);
void mul_backward(Tensor* grad_out, Tensor* out);

Tensor* sum(Tensor* a);
void sum_backward(Tensor* grad_out, Tensor* out);

Tensor* negate(Tensor* a);
void neg_backward(Tensor* grad_out, Tensor* out);

Tensor* divide(Tensor* a, Tensor* b);
void div_backward(Tensor* grad_out, Tensor* out);
