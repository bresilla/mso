#pragma once
#include <vector>
#include <iterator>
#include "perceptron.hpp"

namespace net{
    template < typename Scalar, typename Vector, typename Matrix >
    class Layer {
        // friend class Network;
        private:
            int id;
            int layerSize;
            Scalar biasValue = 0;
            std::vector< Perceptron< Scalar >* > perceptrons;
            Matrix weights;
            Matrix deltas;
            Matrix gradients;
            Vector vals;
            Vector acts;
            Vector ders;
    public:
        int& ID() { return id; }
        Layer( int size, int next = 1, int prev = 1 );
        Perceptron< Scalar >* getPerceptron( int i );
        int& size() { return this->layerSize; }
        Scalar& bias() { return this->biasValue; }
        Matrix& layweights() { return this->weights; }
        Matrix& laydeltas() { return this->deltas; }
        Matrix& laygradients() { return this->gradients; }
        Matrix& VALS();
        Matrix& ACTS( int FUNC = SIGM );
        Matrix& DERS( int FUNC = SIGM );
        void show();
    };

    template < typename Scalar, typename Vector, typename Matrix >
    Layer< Scalar, Vector, Matrix >::Layer(int size, int next, int prev){
        this->size() = size;
        this->VALS() = arma::Col<Scalar>(size);
        this->ACTS() = arma::Col<Scalar>(size);
        this->DERS() = arma::Col<Scalar>(size);
        for (int i = 0; i < size; ++i) {
            Perceptron< Scalar > *n = new Perceptron< Scalar >(0.00);
            n->ID() = i;
            this->perceptrons.push_back(n);
            weights = arma::Mat<Scalar>(size, next, arma::fill::randu);
            deltas = arma::Mat<Scalar>(prev, size, arma::fill::zeros);
        }
    }


    template < typename Scalar, typename Vector, typename Matrix >
    Matrix& Layer< Scalar, Vector, Matrix >::VALS(){
        for (int i = 0; i < this->vals.size(); ++i) {
            this->vals.at(i) = this->perceptrons.at(i)->VAL();
        }
        return this->vals;
    }

    template < typename Scalar, typename Vector, typename Matrix >
    Matrix& Layer< Scalar, Vector, Matrix >::ACTS(int FUNC){
        for (int i = 0; i < this->acts.size(); ++i) {
            this->acts.at(i) = this->perceptrons.at(i)->ACT(FUNC);
        }
        return this->acts;
    }

    template < typename Scalar, typename Vector, typename Matrix >
    Matrix& Layer< Scalar, Vector, Matrix >::DERS(int FUNC){
        for (int i = 0; i < this->ders.size(); ++i) {
            this->ders.at(i) = this->perceptrons.at(i)->DER(FUNC);
        }
        return this->ders;
    }

    template < typename Scalar, typename Vector, typename Matrix >
    Perceptron< Scalar > *Layer< Scalar, Vector, Matrix >::getPerceptron(int i){
        return this->perceptrons.at(i);
    }

    template < typename Scalar, typename Vector, typename Matrix >
    void Layer< Scalar, Vector, Matrix >::show() {
        std::string s = fmt::format( "\t{0} \t {1} \t {2}\n", this->perceptrons.size(), this->gradients.size(), this->weights.size());
        fmt::print( s );    // Python-like format string syntax
    }
}
