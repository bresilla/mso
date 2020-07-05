#include "mso/perceptron.h"

Perceptron::Perceptron(Scalar val){
    this->val = val;
    ACT();
    DER();
}

Scalar& Perceptron::VAL() {
    return this->val;
}

Scalar& Perceptron::ACT(int FUNC){
    switch(FUNC){
        case SIGM: this->act = 1.0 / (1.0 + exp(-this->val)); break;// sigmoid activation
        case TAHN: this->act = std::tanh(this->val); break;// tahn activation
        case RELU: this->act = this->val > 0 ? this->val : this->val * 0.01; break;// LeakyReLU activation
    }
    return this->act;
}

Scalar& Perceptron::DER(int FUNC) {
    switch(FUNC){
        case SIGM: this->der = this->act * (1.0 - this->act); break;// sigmoid derivation
        case TAHN: this->der = 1.0 - std::exp(std::tanh(this->val)); break;// tahn derivation
        case RELU: this->der = this->val >= 0 ? 1.0 : 0.01; break;// LeakyReLU derivation
    }
    return this->der;
}

std::ostream& operator<<(std::ostream& os, Perceptron& n){
    os << "( " << n.VAL() << " , " << n.ACT() << " , " << n.DER() << " )" << std::endl;
    return os;
}
