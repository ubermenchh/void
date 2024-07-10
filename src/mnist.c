#include "void.h"

#define MNIST_TRAIN_SIZE 60000
#define MNIST_TEST_SIZE  10000
#define MNIST_IMAGE_SIZE 784 // 28*28 
#define MNIST_LABEL_SIZE 10 

void read_mnist_data(Tensor** images, Tensor** labels, int num_images) {
    const char* image_file = "../data/train-images.idx3-ubyte";
    const char* label_file = "../data/train-labels.idx1-ubyte"; 
    FILE* f_images = fopen(image_file, "rb");
    FILE* f_labels = fopen(label_file, "rb");

    if (!f_images || !f_labels) {
        fprintf(stderr, "Error opening MNIST files.\n");
        exit(1);
    }
    fseek(f_images, 16, SEEK_SET);
    fseek(f_labels, 8, SEEK_SET);

    *images = tensor_zeros(num_images, MNIST_IMAGE_SIZE, false);
    *labels = tensor_zeros(num_images, MNIST_LABEL_SIZE, false);

   unsigned char pixel, label;
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            fread(&pixel, 1, 1, f_images);
            (*images)->data->data[i * MNIST_IMAGE_SIZE + j] = pixel / 255.0; 
        }
        fread(&label, 1, 1, f_labels);
        for (int j = 0; j < MNIST_LABEL_SIZE; j++) {
            (*labels)->data->data[i * MNIST_LABEL_SIZE + j] = (j == label) ? 1.0 : 0.0;
        }
    } 
    
    fclose(f_images);
    fclose(f_labels);
}

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

int main() {
    Tensor* train_images, *train_labels;
    read_mnist_data(&train_images, &train_labels, 100);

    Module* net = init_mlp(MNIST_IMAGE_SIZE, 256, MNIST_LABEL_SIZE);
    int param_count;
    Tensor** params = net->parameters(net, &param_count);
    Optim* optimizer = init_sgd(params, param_count, 0.01);
    
    int epochs = 2;
    for (int epoch = 0; epoch < epochs; epoch++) {
        Tensor* output = net->forward(net, train_images);
        printf("Output before softmax (first few values): %f, %f, %f\n", 
            output->data->data[0], output->data->data[1], output->data->data[2]);

        Tensor* loss = softmax_cross_entropy(output, train_labels);

        optimizer->zero_grad(optimizer);
        backward(loss);

        for (int i = 0; i < param_count; i++) {
            if (params[i]->grad) {
                float grad_mag = 0;
                for (int j = 0; j < params[i]->grad->data->rows * params[i]->grad->data->cols; j++) {
                    grad_mag += params[i]->grad->data->data[j] * params[i]->grad->data->data[j];
                }
                grad_mag = sqrt(grad_mag);
                printf("Param %d gradient magnitude: %f\n", i, grad_mag);
            }
        }

        optimizer->step(optimizer);
        
        printf("| Epoch: %d | Loss: %f |\n", epoch, loss->data->data[0]);

        free_tensor(output);
        free_tensor(loss);
    }

    free_tensor(train_images);
    free_tensor(train_labels);
    free(params);
    net->free(net);
    optimizer->free(optimizer);
} 
