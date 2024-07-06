#include "void.h"

#define SEED 69

typedef struct {
    Module base;
    Linear* l1;
    Linear* l2;
} MLP;

Tensor* mlp_forward(Module* module, Tensor* input) {
    MLP* mlp = (MLP*)module;

    Tensor* out = linear_forward((Module*)mlp->l1, input);
    out = tensor_relu(out);
    out = linear_forward((Module*)mlp->l2, out);

    return out;
}

MLP* init_mlp(int in_dim, int hidden_dim, int out_dim) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));

    mlp->l1 = init_linear(in_dim, hidden_dim, true);
    mlp->l2 = init_linear(hidden_dim, out_dim, true);

    mlp->base.forward = mlp_forward;
    return mlp;
}

void free_mlp(Module* module) {
    MLP* mlp = (MLP*)module;
    free_linear(mlp->l1);
    free_linear(mlp->l2);
    free(mlp);
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
    
    MLP* net = init_mlp(2, 4, 1);
    int param_count1, param_count2;
    Tensor** params1 = net->l1->base.parameters((Module*)net->l1, &param_count1);
    Tensor** params2 = net->l2->base.parameters((Module*)net->l2, &param_count2);
    Tensor** all_params = (Tensor**)malloc(sizeof(Tensor*) * (param_count1 + param_count2));
    for (int i = 0; i < param_count1; i++) {
        all_params[i] = params1[i];
    } 
    for (int i = 0; i < param_count2; i++) {
        all_params[i + param_count1] = params2[i];
    }
    SGD* optimizer = init_sgd(all_params, param_count1 + param_count2, 0.01);
    int epochs = 20;
    for (int epoch = 0; epoch < epochs; epoch++) {
        Tensor* y_pred = net->base.forward((Module*)net, x_train);
        Tensor* loss = mse(y_train, y_pred);
        backward(loss);
        optimizer->base.step(optimizer);
        optimizer->base.zero_grad(optimizer);

        printf("Epoch: %d, Loss: %f\n", epoch, loss->data->data[0]);

        if (epoch == epochs-1) print_tensor(y_pred);
        free_tensor(y_pred);
        free_tensor(loss);
    } 

    free_tensor(x_train);
    free_tensor(y_train);
    free(all_params);
    free_sgd(optimizer);

    return 0;
}
