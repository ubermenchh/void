#include "void.h"

#define SEED 1337

typedef struct {
    Module base;
    Linear* l1;
    Linear* l2;
} MLP;

Tensor* mlp_forward(Module* module, Tensor* input) {
    MLP* mlp = (MLP*)module->impl;
    Tensor* out = mlp->l1->base.forward((Module*)mlp->l1, input);
    out = tensor_relu(out);
    out = mlp->l2->base.forward((Module*)mlp->l2, out);

    return out;
}

void free_mlp(Module* module) {
    MLP* mlp = (MLP*)module->impl;
    mlp->l1->base.free((Module*)mlp->l1);
    mlp->l2->base.free((Module*)mlp->l2);
    free(mlp);
}

Tensor** mlp_parameters(Module* module, int* count) {
    MLP* mlp = (MLP*)module->impl;
    int count1, count2;
    Tensor** params1 = mlp->l1->base.parameters((Module*)mlp->l1, &count1);
    Tensor** params2 = mlp->l2->base.parameters((Module*)mlp->l2, &count2);

    *count = count1 + count2;
    Tensor** all_params = malloc(*count * sizeof(Tensor*));

    for (int i = 0; i < count1; i++) all_params[i] = params1[i];
    for (int i = 0; i < count2; i++) all_params[i + count1] = params2[i];

    free_tensor(*params1);
    free_tensor(*params2);

    return all_params;
}

Module* init_mlp(int in_dim, int hidden_dim, int out_dim) {
    MLP* mlp = malloc(sizeof(MLP));
    mlp->base.impl = mlp;
    mlp->base.forward = mlp_forward;
    mlp->base.parameters = mlp_parameters;
    mlp->base.free = free_mlp;

    mlp->l1 = (Linear*)init_linear(in_dim, hidden_dim, true);
    mlp->l2 = (Linear*)init_linear(hidden_dim, out_dim, true);
    
    return (Module*)mlp;
}

float compute_numerical_gradient(Tensor* param, Tensor* (*loss_func)(Tensor*, Tensor*), Tensor* y_true, Tensor* y_pred, Module* net, float epsilon) {
    float original_value = param->data->data[0];
    
    param->data->data[0] = original_value + epsilon;
    Tensor* output_plus = net->forward(net, y_pred);
    Tensor* loss_plus = loss_func(y_true, output_plus);
    float loss_plus_value = loss_plus->data->data[0];
    
    param->data->data[0] = original_value - epsilon;
    Tensor* output_minus = net->forward(net, y_pred);
    Tensor* loss_minus = loss_func(y_true, output_minus);
    float loss_minus_value = loss_minus->data->data[0];
    
    param->data->data[0] = original_value;
    
    float numerical_grad = (loss_plus_value - loss_minus_value) / (2 * epsilon);
    
    free_tensor(output_plus);
    free_tensor(output_minus);
    free_tensor(loss_plus);
    free_tensor(loss_minus);
    
    return numerical_grad;
}

int main() {
    double x_train_data[] = {
        0, 0,
        0, 1,
        1, 0, 
        1, 1
    };
    double y_train_data[] = {
        0,
        1,
        1, 
        0
    };
    Matrix* data_x = InitMatrix(4, 2);
    SetElements(data_x, x_train_data);
    Matrix* data_y = InitMatrix(4, 1);
    SetElements(data_y, y_train_data);
    Tensor* x_train = init_tensor(data_x, true);
    Tensor* y_train = init_tensor(data_y, true);
    
    Module* net = init_mlp(2, 6, 1);
    int param_count;
    Tensor** params = net->parameters(net, &param_count);
    Optim* optimizer = init_sgd(params, param_count, 0.02);
    
    int epochs = 100;
    for (int epoch = 0; epoch < epochs; epoch++) {
        Tensor* y_pred = net->forward(net, x_train);
        Tensor* loss = mse(y_train, y_pred);

        optimizer->zero_grad(optimizer);
        backward(loss);
        optimizer->step(optimizer);

        printf("Epoch: %d, Loss: %f\n", epoch, loss->data->data[0]);

        if (epoch == epochs-1) print_tensor(y_pred);

        free_tensor(y_pred);
        free_tensor(loss);
    } 


    free_tensor(x_train);
    free_tensor(y_train);
    free(params);
    net->free(net);
    optimizer->free(optimizer);

    return 0;
}
