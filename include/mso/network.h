#pragma once
#include <math.h>
#include <cassert>
#include "mso/layer.h"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#define CLASS 1
#define REGRS 2

class Network{
    private:
        int networkDepth;
        int activationFunc = SIGM;
        int lossFunc = CLASS;
        std::vector<Layer *> layers;
        std::vector<int> topology;

        std::vector<double> input;
        std::vector<double> truth;
        std::vector<double> output;

        double cost;
        double eta=0.1, alpha=0.1, biasval = 0.0;
        int trainepochs;
        int currentepoch;
        arma::vec loss;
        std::vector<double> histerr;

        double calcError(double predict, double truth);
        double calcCost();


    public:
        Network(std::vector<int> topology);
        Layer *getLayer(int i);

        int& depth() {return this->networkDepth;}
        int& activation() {return this->activationFunc;}
        int& lossfunction() {return this->lossFunc;}
        int& epochs() {return this->trainepochs;}
        // double& momentum(){ return this->alpha = std::pow(trainepochs,-(currentepoch/10));}
        double& momentum(){ return this->alpha;}
        double& learnrate(){return this->eta;}
        double& bias(){return this->biasval;}
        void setInput(std::vector<double> input);
        void setTruth(std::vector<double> truth);
        bool roundOut = false;
        std::string weightPath;

        void forewardProp();
        void backwardProp();
        void train(std::vector<std::vector<double>> train, std::vector<std::vector<double>> labels);
        void test(std::vector<std::vector<double>> test);
        void saveWeights();
        void loadWeights();

        void SHAPE(arma::mat mat, std::string text = "");
        void showTopology();
        friend std::ostream& operator<<(std::ostream& os, Network& n);
};
