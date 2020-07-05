#pragma once
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <armadillo>

// #include <libtorch/include/torch/csrc/api/include/torch/torch.h>

typedef double Scalar;

#define SIGM 1
#define TAHN 2
#define RELU 3

#define LOG(x) std::cout << std::setprecision(32) << x << std::endl;

class Perceptron {
    friend class Layer;
    private:
        Scalar val;
        Scalar act;
        Scalar der;
        int id;

    public:
        Perceptron(Scalar val);
        Scalar& VAL();
        Scalar& ACT(int FUNC = SIGM);
        Scalar& DER(int FUNC = SIGM);
        int& ID(){return id;}

        friend std::ostream& operator<<(std::ostream& os, Perceptron& m);
};
