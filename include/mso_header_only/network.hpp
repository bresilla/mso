#pragma once
#include "layer.hpp"
#include <cassert>
#include <math.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#define CLASS 1
#define REGRS 2

namespace net {

    template < typename Scalar, typename Vector, typename Matrix >
    class Network {
    private:
        int networkDepth;
        int activationFunc = SIGM;
        int lossFunc = CLASS;
        std::vector< Layer< Scalar, Vector, Matrix >* > layers;
        std::vector< int > topology;

        std::vector< double > input;
        std::vector< double > truth;
        std::vector< double > output;

        double cost;
        double eta = 0.1, alpha = 0.1, biasval = 0.0;
        int trainepochs;
        int currentepoch;
        arma::vec loss;
        std::vector< double > histerr;

        double calcError( double predict, double truth );
        double calcCost();

    public:
        Network( std::vector< int > topology );
        Layer< Scalar, Vector, Matrix >* getLayer( int i );

        int& depth() { return this->networkDepth; }
        int& activation() { return this->activationFunc; }
        int& lossfunction() { return this->lossFunc; }
        int& epochs() { return this->trainepochs; }
        // double& momentum(){ return this->alpha = std::pow(trainepochs,-(currentepoch/10));}
        double& momentum() { return this->alpha; }
        double& learnrate() { return this->eta; }
        double& bias() { return this->biasval; }
        void setInput( std::vector< double > input );
        void setTruth( std::vector< double > truth );
        bool roundOut = false;
        std::string weightPath;

        void forewardProp();
        void backwardProp();
        void train( std::vector< std::vector< double > > train, std::vector< std::vector< double > > labels );
        void test( std::vector< std::vector< double > > test );
        void saveWeights();
        void loadWeights();

        void SHAPE( arma::mat mat, std::string text = "" );
        void showTopology();
        void show();
    };

    template < typename Scalar, typename Vector, typename Matrix >
    Network< Scalar, Vector, Matrix >::Network( std::vector< int > topology ) {
        this->topology = topology;
        this->depth() = topology.size();
        int i = 0;
        for( i = 0; i < depth(); ++i ) {
            int next = i < depth() - 1 ? topology.at( i + 1 ) : 1;
            int prev = i != 0 ? topology.at( i - 1 ) : 1;
            Layer< Scalar, Vector, Matrix >* l = new Layer< Scalar, Vector, Matrix >( topology.at( i ), next, prev );
            l->ID() = i;
            this->layers.push_back( l );
        }
    }

    template < typename Scalar, typename Vector, typename Matrix >
    Layer< Scalar, Vector, Matrix >* Network< Scalar, Vector, Matrix >::getLayer( int i ) {
        return this->layers.at( i );
    }

    template < typename Scalar, typename Vector, typename Matrix >
    void Network< Scalar, Vector, Matrix >::setInput( std::vector< double > input ) {
        this->input = std::vector< double >( input.begin(), input.begin() + this->layers.front()->size() );
        if( this->input.size() == 0 || this->input.size() != this->layers.front()->size() ) { assert( false ); }
        for( int i = 0; i < this->layers.front()->size(); ++i ) { layers.front()->getPerceptron( i )->VAL() = input.at( i ); }
    }

    template < typename Scalar, typename Vector, typename Matrix >
    void Network< Scalar, Vector, Matrix >::setTruth( std::vector< double > truth ) {
        this->truth = std::vector< double >( truth.begin(), truth.begin() + this->layers.back()->size() );
        if( this->truth.size() == 0 || this->truth.size() != this->layers.back()->size() ) { assert( false ); }
        this->output = std::vector< double >( this->truth );
    }
    template < typename Scalar, typename Vector, typename Matrix >
    void Network< Scalar, Vector, Matrix >::showTopology() {
        std::cout << std::endl
                  << "LAYERS: ";
        for( int i = 0; i < this->topology.size(); ++i ) {
            std::cout << topology.at( i ) << ", ";
        }
        std::cout << std::endl;
    }

    template < typename Scalar, typename Vector, typename Matrix >
    void Network< Scalar, Vector, Matrix >::SHAPE( arma::mat mat, std::string text ) {
        std::cout << text << " --> (" << mat.n_cols << " " << mat.n_rows << ")" << std::endl;
    }

    template < typename Scalar, typename Vector, typename Matrix >
    double Network< Scalar, Vector, Matrix >::calcError( double predict, double truth ) {
        double val;
        switch( lossFunc ) {
            case CLASS: val = predict - truth; break;
            case REGRS: val = -( truth * log( predict ) + ( 1 - truth ) * log( 1 - predict ) ); break;    //logarithmic loss
        }
        return val;
    }

    template < typename Scalar, typename Vector, typename Matrix >
    double Network< Scalar, Vector, Matrix >::calcCost() {
        this->cost = 0.00;
        for( auto i : this->loss ) { this->cost += i; }
        this->cost /= this->loss.size();
        return this->cost;
    }

    template < typename Scalar, typename Vector, typename Matrix >
    void Network< Scalar, Vector, Matrix >::forewardProp() {
        this->loss = std::vector< double >( this->layers.back()->size() );
        for( int i = 0; i < this->depth() - 1; ++i ) {
            arma::vec input = i == 0 ? this->layers.at( i )->VALS() : this->layers.at( i )->ACTS( activationFunc );
            arma::mat weight = this->layers.at( i )->layweights();
            arma::mat result = arma::Mat< double >();
            result = ( input.t() * weight ) + this->bias();
            for( int j = 0; j < this->layers.at( i + 1 )->size(); ++j ) {
                this->layers.at( i + 1 )->getPerceptron( j )->VAL() = result.at( j );
            }
        }
    }

    template < typename Scalar, typename Vector, typename Matrix >
    void Network< Scalar, Vector, Matrix >::backwardProp() {
        for( int i = 0; i < this->layers.back()->size(); ++i ) {
            auto predict = this->layers.back()->getPerceptron( i )->ACT( activationFunc );
            auto truth = this->truth.at( i );
            this->loss.at( i ) = calcError( predict, truth );
            this->output.at( i ) = this->roundOut ? round( predict ) : predict;
        }
        calcCost();
        this->histerr.push_back( this->cost );

        for( int i = this->depth() - 1; i > 0; --i ) {
            auto& grad = this->layers.at( i )->laygradients();
            auto oldeltas = this->layers.at( i )->laydeltas();
            if( i == this->depth() - 1 ) {
                grad = this->loss % this->layers.back()->DERS( activationFunc );
            } else {
                grad = ( this->layers.at( i + 1 )->laygradients().t() * this->layers.at( i )->layweights().t() ).t() % this->layers.at( i )->DERS( activationFunc );
            }
            this->layers.at( i )->laydeltas() = ( grad * this->layers.at( i - 1 )->ACTS( activationFunc ).t() ).t();
            this->layers.at( i )->laydeltas() = this->layers.at( i )->laydeltas() * eta + ( alpha * oldeltas );
            this->layers.at( i - 1 )->layweights() -= this->layers.at( i )->laydeltas();
        }
    }

    template < typename Scalar, typename Vector, typename Matrix >
    void Network< Scalar, Vector, Matrix >::train( std::vector< std::vector< double > > train, std::vector< std::vector< double > > labels ) {
        if( train.size() != labels.size() ) { assert( false ); }
        for( int i = 0; i < train.size(); ++i ) {
            std::cout << std::endl
                      << "#################### SAMPLE: " << i << " ####################" << std::endl;
            this->setInput( train.at( i ) );
            this->setTruth( labels.at( i ) );
            for( int i = 1; i <= this->epochs(); ++i ) {
                this->currentepoch = i;
                forewardProp();
                backwardProp();
                momentum();
            }
            for( int i = 0; i < this->layers.back()->size(); ++i ) {
                std::cout << this->output.at( i ) << std::endl;
            }
            this->saveWeights();
        }
    }

    template < typename Scalar, typename Vector, typename Matrix >
    void Network< Scalar, Vector, Matrix >::test( std::vector< std::vector< double > > test ) {
        for( int i = 0; i < test.size(); ++i ) {
            std::cout << std::endl
                      << "#################### TEST: " << i << " ####################" << std::endl;
            this->setInput( test.at( i ) );
            this->loadWeights();
            forewardProp();
            for( int j = 0; j < this->layers.back()->size(); ++j ) {
                std::cout << this->layers.back()->perceptrons.at( j )->ACT( activationFunc ) << std::endl;
            }
        }
    }

    template < typename Scalar, typename Vector, typename Matrix >
    void Network< Scalar, Vector, Matrix >::saveWeights() {
        json weightJSON = {};
        std::vector< std::vector< std::vector< double > > > weights;
        for( int i = 0; i < this->depth(); ++i ) {
            std::vector< std::vector< double > > rows;
            for( int j = 0; j < this->layers.at( i )->size(); ++j ) {
                auto b = arma::conv_to< std::vector< double > >::from( this->layers.at( i )->layweights().row( j ) );
                rows.push_back( b );
            }
            weights.push_back( rows );
        }
        weightJSON["weights"] = weights;
        std::ofstream tempfile( weightPath );
        tempfile << std::setw( 4 ) << weightJSON << std::endl;
    }

    template < typename Scalar, typename Vector, typename Matrix >
    void Network< Scalar, Vector, Matrix >::loadWeights() {
        std::ifstream ifs( weightPath );
        json weightJSON;
        ifs >> weightJSON;
        std::vector< std::vector< std::vector< double > > > weights = weightJSON["weights"];
        for( int i = 0; i < this->depth(); ++i ) {
            for( int j = 0; j < weights.at( i ).size(); ++j ) {
                for( int k = 0; k < weights.at( i ).at( j ).size(); ++k ) {
                    this->layers.at( i )->layweights().at( j, k ) = weights.at( i ).at( j ).at( k );
                }
            }
        }
    }

    template < typename Scalar, typename Vector, typename Matrix >
    void Network< Scalar, Vector, Matrix >::show() {
        for( int i = 0; i < this->depth(); ++i ) {
            if( i == this->depth() - 1 ) {
                fmt::print( "------- OUTPUT LAYER SIZE: {0} ------", this->layers.at( i )->size() );
            } else if( i == 0 ) {
                fmt::print( "------- INPUT LAYER SIZE: {0} -------", this->layers.at( i )->size() );
            } else {
                fmt::print( "----- HIDEN LAYER ({0}) SIZE: {1}", this->layers.at( i )->ID(), this->layers.at( i )->size() );
            }
            for( int j = 0; j < this->layers.at( i )->size(); ++j ) {
                auto first = i == 0 ? this->layers.at( i )->getPerceptron( j )->VAL() : this->layers.at( i )->getPerceptron( j )->ACT( this->activationFunc );
                fmt::print( "V:  {0} \tW:  {1}", first, this->layers.at( i )->layweights().row( j ) );
            }
        }
    }
}

// std::ostream& operator<<(std::ostream& os, Network& n){
// for (int i = 0; i < n.depth(); ++i) {
// if (i == n.depth()-1) {os << std::endl << "------- OUTPUT LAYER SIZE: " << n.layers.at(i)->size() << " ------" << std::endl;}
// else if (i == 0) {os << std::endl << "------- INPUT LAYER SIZE: " << n.layers.at(i)->size() << " -------" << std::endl;}
// else {os << std::endl << "----- HIDEN LAYER ("<< n.layers.at(i)->ID() <<") SIZE: " << n.layers.at(i)->size() << " -----" << std::endl;}
// for (int j = 0; j < n.layers.at(i)->size(); ++j) {
// auto first = i==0 ? n.layers.at(i)->getPerceptron(j)->VAL() : n.layers.at(i)->getPerceptron(j)->ACT(n.activationFunc);
// os << "V: " << first << "\tW:"<< n.layers.at(i)->layweights().row(j);
// }
// }
// return os;
// }
