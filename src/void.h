#include <flash.h>

typedef struct Neuron {
    Matrix* weights;
    Matrix* bias;
    
    Matrix* (*forward)  (struct Neuron*, Matrix**);
    void    (*backward) (struct Neuron*, Matrix*, Matrix*, Matrix*, double);
} Neuron;

typedef struct NN {
    Neuron* in;  // input layer
    Neuron* hd;  // hidden layer 
    Neuron* out; // output layer

    Matrix* (*forward)  (struct NN*, Matrix**);
    void    (*backward) (struct NN*, Matrix*, Matrix*, double);
} NN;

// Neuron Function
Neuron* InitNeuron(int in_dim, int out_dim, bool bias);
void PrintNeuron(Neuron* n);
void FreeNeuron(Neuron* n);
Matrix* NeuronForward(Neuron*, Matrix** in);
void NeuronBackward(Neuron*, Matrix* in, Matrix* pred, Matrix* targ, double lr);

// Neural Network (NN) Functions
NN* InitNN(int in_dim, int hidden_dim, int out_dim, bool bias);
void FreeNN(NN* net);
Matrix* NNForward(NN* net, Matrix** in);
void NNBackward(NN* net, Matrix* in, Matrix* targ, double lr);

// Loss Functions
double MSE(Matrix* y_pred, Matrix* y_true);
Matrix* MSEBackward(Matrix* y_pred, Matrix* y_true);

// Activation Functions 
double sigmoid(double x);
Matrix* MatrixSigmoid(Matrix* m);
Matrix* MatrixSigmoidBackward(Matrix* m);
