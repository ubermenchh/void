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
    ctx->saved_tensors = tensors;
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
        t->grad = grad;
    } else {
        for (int i = 0; i < t->data->rows * t->data->cols; i++) {
            t->grad->data[i] += grad->data[i];
        }
    }
    if (t->grad_fn != NULL) {
        Tensor grad_output;
        grad_output.data = grad;
        t->grad_fn(&grad_output, t);
    }
}

Tensor* add(Tensor* a, Tensor* b) {
    Tensor* out = init_tensor(MatrixAdd(a->data, b->data), a->requires_grad || b->requires_grad);

    if (out->requires_grad) {
        out->ctx = (Context*)malloc(sizeof(Context));
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
        if (a->grad == NULL) {
            a->grad = InitMatrix(a->data->rows, a->data->cols);
        }
        for (int i = 0; i < a->data->rows*a->data->cols; i++) {
            a->grad->data[i] += grad_output->data->data[i];
        }
    }
    if (b->requires_grad) {
        if (b->grad == NULL) {
            b->grad = InitMatrix(b->data->rows, b->data->cols);
        }
        for (int i = 0; i < b->data->rows*b->data->cols; i++) {
            b->grad->data[i] += grad_output->data->data[i];
        }
    }
}

Tensor* mul(Tensor* a, Tensor* b) {
    Tensor* out = init_tensor(
        MatrixDotProduct(a->data, b->data), 
        a->requires_grad || b->requires_grad
    );

    if (out->requires_grad) {
        out->ctx = (Context*)malloc(sizeof(Context));
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
        if (a->grad == NULL) {
            a->grad = InitMatrix(a->data->rows, a->data->cols);
        }
        for (int i = 0; i < a->data->rows*a->data->cols; i++) {
            a->grad->data[i] += grad_output->data->data[i] * b->data->data[i];
        }
    }
    if (b->requires_grad) {
        if (b->grad == NULL) {
            b->grad = InitMatrix(b->data->rows, b->data->cols);
        }
        for (int i = 0; i < b->data->rows*b->data->cols; i++) {
            b->grad->data[i] += grad_output->data->data[i] * a->data->data[i];
        }
    }
}
