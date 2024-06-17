#include "void.h"

#define RAND_SEED 1337

// NEURON

Neuron* InitNeuron(int in_dim, int out_dim, bool bias) {
    assert(in_dim > 0 && out_dim > 0);
    Neuron* n = (Neuron*)malloc(sizeof(Neuron));
    if (n == NULL) {
        return NULL;
    }

    // X * 2 - 1
    n->weights = MatrixScalarSub(MatrixScalarMul(RandnMatrix(out_dim, in_dim, RAND_SEED), 2), 1);
    if (bias) {
        n->bias = MatrixScalarSub(MatrixScalarMul(RandnMatrix(out_dim, 1, RAND_SEED), 2), 1);
    } else {
        n->bias = ZerosMatrix(out_dim, 1);
    }

    n->forward = NeuronForward;
    n->backward = NeuronBackward;

    return n;
}

void FreeNeuron(Neuron* n) {
    FreeMatrix(n->weights);
    FreeMatrix(n->bias);
    free(n);
}

void PrintNeuron(Neuron* n) {
    printf("Weights:\n");
    PrintMatrix(n->weights);
    printf("Bias:\n");
    PrintMatrix(n->bias);
}

Matrix* NeuronForward(Neuron* n, Matrix** in) {
    // x*W + B 
    Matrix* z1 = MatrixMul(*in, n->weights);
    Matrix* z2 = MatrixAdd(z1, n->bias);
    
    FreeMatrix(z1);

    return z2;
}

void NeuronBackward(Neuron* n, Matrix* in, Matrix* pred, Matrix* targ, double lr) {
    Matrix* dloss = MSEBackward(pred, targ);
    Matrix* dw = MatrixDotProduct(MatrixTranspose(in), dloss);
    Matrix* db = MatrixSumVals(dloss, 0);

    Matrix* new_weights = MatrixSub(n->weights, MatrixScalarMul(dw, lr));
    Matrix* new_bias    = MatrixSub(n->bias   , MatrixScalarMul(db, lr));

    FreeMatrix(new_bias);
    FreeMatrix(new_weights);
    
    n->weights = new_weights;
    n->bias    = new_bias;

    FreeMatrix(db);
    FreeMatrix(dw);
    FreeMatrix(dloss);
}   

// NEURAL NETWORK (NN)
NN* InitNN(int in_dim, int hidden_dim, int out_dim, bool bias) {
    NN* net = (NN*)malloc(sizeof(NN));
    if (net == NULL) {
        return NULL;
    }

    net->in  = InitNeuron(in_dim, hidden_dim, bias);
    net->hd  = InitNeuron(hidden_dim, hidden_dim, bias);
    net->out = InitNeuron(hidden_dim, out_dim, bias);

    net->forward = NNForward;
    net->backward = NNBackward;

    return net;
}

void FreeNN(NN* net) {
    FreeNeuron(net->in);
    FreeNeuron(net->hd);
    FreeNeuron(net->out);
    free(net);
}

Matrix* NNForward(NN* net, Matrix** in) {
    Matrix* z1 = net->in->forward(net->in, in);
    Matrix* a1 = MatrixSigmoid(z1);

    Matrix* z2 = net->hd->forward(net->hd, &a1);
    Matrix* a2 = MatrixSigmoid(z2);

    Matrix* out = net->out->forward(net->out, &a2);

    FreeMatrix(z1);
    FreeMatrix(a1);
    FreeMatrix(z2);
    FreeMatrix(a2);

    return out;
} 

void NNBackward(NN* net, Matrix* in, Matrix* targ, double lr) {
    // Forward
    Matrix* z1 = net->in->forward(net->in, &in);
    Matrix* a1 = MatrixSigmoid(z1);

    Matrix* z2 = net->hd->forward(net->hd, &a1);
    Matrix* a2 = MatrixSigmoid(z2);

    Matrix* out = net->out->forward(net->out, &a2); 
    
    // Backward 
    Matrix* dout = MSEBackward(out, targ);
    net->out->backward(net->out, in, dout, a2, lr);
    
    PrintMatrix(net->out->weights);
    PrintMatrix(dout);
    PrintMatrix(MatrixMul(net->out->weights, dout));
    Matrix* da2 = MatrixMul(dout, MatrixTranspose(net->out->weights));
    Matrix* dz2 = MatrixSigmoidBackward(da2, a2);
    net->hd->backward(net->hd, in, dz2, a1, lr);

    Matrix* da1 = MatrixMul(dz2, MatrixTranspose(net->hd->weights));
    Matrix* dz1 = MatrixSigmoidBackward(da1, a1);
    net->in->backward(net->in, in, dz1, in, lr);

    // Freeing the allocated memory
    FreeMatrix(z1);
    FreeMatrix(a1);
    FreeMatrix(z2);
    FreeMatrix(a2);
    FreeMatrix(out);
    FreeMatrix(dout);
    FreeMatrix(da2);
    FreeMatrix(dz2);
    FreeMatrix(da1);
    FreeMatrix(dz1);
}

Matrix* NNTrain(NN* net, Matrix** in, Matrix** targ, bool grad) {
    return NULL;
}

// LOSS FUNCTION
double MSE(Matrix* y_pred, Matrix* y_true) {
    assert(y_pred->rows == y_true->rows);
    assert(y_pred->cols == y_true->cols);
    // out = (y_true - y_pred)^2
    Matrix* diff = MatrixSub(y_pred, y_true);  // (y_pred - y_true)
    Matrix* sq   = MatrixPower(diff, 2);       // (y_pred - y_true)**2
    double sum   = MatrixSum(sq);              // sum((y_pred - y_true)**2)

    FreeMatrix(diff);
    FreeMatrix(sq);

    return sum / y_pred->rows;                 // (1/N) * sum((y_pred - y_true)**2)
}

Matrix* MSEBackward(Matrix* y_pred, Matrix* y_true) {
    // (2/N) * (y_pred - y_true)
    Matrix* diff = MatrixSub(y_pred, y_true);           // (y_pred - y_true)          
    Matrix* dbl  = MatrixScalarMul(diff, 2.0);          // 2 * (y_pred - y_true)
    Matrix* grad = MatrixScalarDiv(dbl, y_pred->rows);  // (2/N) * (y_pred - y_true)
    
    FreeMatrix(dbl);
    FreeMatrix(diff);

    return grad;
}

// ACTIVATION FUNCTION
double sigmoid(double x) {
    // 1 / 1 + e^(-x)
    return 1.0 / (1.0 + exp(-x));
}

Matrix* MatrixSigmoid(Matrix* m) {
    Matrix* out = InitMatrix(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(out, i, j) = sigmoid(MAT_AT(m, i, j));
        }
    }
    return out;
}

Matrix* MatrixSigmoidBackward(Matrix* m) {
    Matrix* grad = InitMatrix(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            MAT_AT(grad, i, j) = sigmoid(MAT_AT(m, i, j)) * (1 - sigmoid(MAT_AT(m, i, j)));
        }
    }
    return grad;
}
