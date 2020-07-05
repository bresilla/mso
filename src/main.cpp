#include "mso.hpp"
#include "mso/network.h"
#include <iostream>

using namespace arma;
typedef double Scalar;
typedef arma::vec Vector;
typedef arma::mat Matrix;

int main( int argc, char** argv ) {
    std::vector< int > topology = { 3, 4, 1 };
    // auto netconfig = utl::jsonParser("/home/bresilla/DATA/CODE/PROJECTS/OTHER/mso/etc/config/netconfig.json");
    // double momentum = netconfig["momentum"];
    // double learnrate = netconfig["learnrate"];
    // double bias = netconfig["bias"];
    // int activation = netconfig["activation"];
    // int lossfunction = netconfig["lossfunction"];
    // int epochs = netconfig["epochs"];

    // auto labels = utl::fetchCSV(netconfig["labels"]);
    // auto train = utl::fetchCSV(netconfig["train"]);
    // auto test = utl::fetchCSV(netconfig["test"]);
    // auto weights = netconfig["weights"];
    // std::vector<int> topology = netconfig["topology"];

    Network* n = new Network( topology );
    std::cout << *n << std::endl;
    Perceptron* p = new Perceptron( 5 );
    Layer* l = new Layer( 5, 2, 6 );

    // n->weightPath = weights;
    // n->momentum() = momentum;
    // n->learnrate() = learnrate;
    // n->bias() = bias;
    // n->activation() = activation;
    // n->lossfunction() = lossfunction;
    // n->epochs() = epochs;
    // // n->roundOut = true;
    // n->showTopology();

    // n->train(train, labels);
    // n->test(test);
    // // n->loadWeights();

    auto* np = new net::Perceptron< Scalar >( 5 );
    np->show();

    auto* nl = new net::Layer< Scalar, Vector, Matrix >( 5, 2, 6 );
    nl->show();

    auto* nn = new net::Network< Scalar, Vector, Matrix >( topology );
    // n->show();

    return 0;
}
