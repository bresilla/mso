#include "mso/layer.h"

Layer::Layer(int size, int next, int prev){
    this->size() = size;
    this->VALS() = arma::Col<double>(size);
    this->ACTS() = arma::Col<double>(size);
    this->DERS() = arma::Col<double>(size);
    for (int i = 0; i < size; ++i) {
        Perceptron *n = new Perceptron(0.00);
        n->ID() = i;
        this->perceptrons.push_back(n);
        weights = arma::Mat<double>(size, next, arma::fill::randu);
        deltas = arma::Mat<double>(prev, size, arma::fill::zeros);
    }
}


arma::mat& Layer::VALS(){
    for (int i = 0; i < this->vals.size(); ++i) {
        this->vals.at(i) = this->perceptrons.at(i)->VAL();
    }
    return this->vals;
}
arma::mat& Layer::ACTS(int FUNC){
    for (int i = 0; i < this->acts.size(); ++i) {
        this->acts.at(i) = this->perceptrons.at(i)->ACT(FUNC);
    }
    return this->acts;
}

arma::mat& Layer::DERS(int FUNC){
    for (int i = 0; i < this->ders.size(); ++i) {
        this->ders.at(i) = this->perceptrons.at(i)->DER(FUNC);
    }
    return this->ders;
}

Perceptron *Layer::getPerceptron(int i){
    return this->perceptrons.at(i);
}

std::ostream& operator<<(std::ostream& os, Layer& l){
    os << "\tPerceptrons: " << l.perceptrons.size() << "\t\tWeight(IN): " << l.gradients.size() << "\t\tWeight(OUT): " << l.weights.size();
    os << std::endl;
    return os;
}
