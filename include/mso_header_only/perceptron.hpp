#pragma once
#include <algorithm>
#include <iomanip>
#include <iostream>

#include <armadillo>
#include <fmt/format.h>

#define SIGM 1
#define TAHN 2
#define RELU 3

namespace net{
    template < typename Scalar >
    class Perceptron {
        // friend class Layer;
        private:
            Scalar val;
            Scalar act;
            Scalar der;
            int id;
        public:
            Perceptron( Scalar val );
            Scalar& VAL();
            Scalar& ACT( int FUNC = SIGM );
            Scalar& DER( int FUNC = SIGM );
            int& ID() { return id; }
            void show();
    };

    template < typename Scalar >
    Perceptron< Scalar >::Perceptron( Scalar val ) {
        this->val = val;
        ACT();
        DER();
    }
    template < typename Scalar >
    Scalar& Perceptron< Scalar >::VAL() {
        return this->val;
    }
    template < typename Scalar >
    Scalar& Perceptron< Scalar >::ACT( int FUNC ) {
        switch( FUNC ) {
            case SIGM: this->act = 1.0 / ( 1.0 + exp( -this->val ) ); break;               // sigmoid activation
            case TAHN: this->act = std::tanh( this->val ); break;                          // tahn activation
            case RELU: this->act = this->val > 0 ? this->val : this->val * 0.01; break;    // LeakyReLU activation
        }
        return this->act;
    }
    template < typename Scalar >
    Scalar& Perceptron< Scalar >::DER( int FUNC ) {
        switch( FUNC ) {
            case SIGM: this->der = this->act * ( 1.0 - this->act ); break;             // sigmoid derivation
            case TAHN: this->der = 1.0 - std::exp( std::tanh( this->val ) ); break;    // tahn derivation
            case RELU: this->der = this->val >= 0 ? 1.0 : 0.01; break;                 // LeakyReLU derivation
        }
        return this->der;
    }
    template < typename Scalar >
    void Perceptron< Scalar >::show() {
        std::string s = fmt::format( "\t{0} \t {1} \t {2}\n", this->VAL(), this->ACT(), this->DER() );
        fmt::print( s );    // Python-like format string syntax
    }
}
