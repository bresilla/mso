#pragma once
#include <vector>
#include <iterator>
#include "mso/perceptron.h"

class Layer {
    friend class Network;
    private:
    int id;
        int layerSize;
        Scalar biasValue = 0;

        std::vector<Perceptron *> perceptrons;

        arma::mat weights;
        arma::mat deltas;
        arma::mat gradients;

        arma::vec vals;
        arma::vec acts;
        arma::vec ders;
    public:
        int& ID(){return id;}
        Layer(int size, int next = 1, int prev = 1);
        Perceptron *getPerceptron(int i);

        int& size() {return this->layerSize;}
        Scalar& bias(){return this->biasValue;}
        arma::mat& layweights(){return this->weights;}
        arma::mat& laydeltas(){return this->deltas;}
        arma::mat& laygradients(){return this->gradients;}

        arma::mat& VALS();
        arma::mat& ACTS(int FUNC = SIGM);
        arma::mat& DERS(int FUNC = SIGM);

        friend std::ostream& operator<<(std::ostream& os, Layer& l);
        // std::vector<Perceptron *>* begin() {return &perceptrons;}
        // std::vector<Perceptron *>* end() {return &perceptrons+sizeof(perceptrons);}
};
