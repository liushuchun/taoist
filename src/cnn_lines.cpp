//
// Created by Shuchun on 2017/12/19.
//

#include "cnn_lines.h"

#include <sstream>

using namespace std;

namespace cnn {


    bool ParameterLine::has_parameters() const { return true; }

    string ParameterLine::to_string(const vector<string> &names) const {
        ostringstream s;
        s << "params" << tensor;
        return s.str();

    }

    Matrix ParameterLine::forward(const std::vector<const Matrix *> &xs) const {
        assert(xs.size() == 0);
        return values;
    }

    Matrix
    ParameterLine::backward(const vector<const Matrix *> &xs, const Matrix &fx, const Matrix &dedf, unsigned i) const {
        return Matrix();
    }

    string InputLine::to_string(const std::vector<std::string> &names) const {
        ostringstream s;
        s << "inputs" << tensor;
        return s.str();
    }

    Matrix InputLine::forward(const std::vector<const Matrix *> &xs) const {
        assert(xs.size() == 0);
        return values;
    }

    Matrix InputLine::backward(const std::vector<const Matrix *> &xs, const Matrix &fx, const Matrix &dedf,
                               unsigned i) const {
        return Matrix();
    }

    string MatrixMultiply::to_string(const std::vector<std::string> &args) const {
        ostringstream s;
        s << args[0] << " * " << args[1];
        return s.str();
    }

    Matrix MatrixMultiply::forward(const std::vector<const Matrix *> &xs) const {
        assert(xs.size() == 2);
        return (*xs[0]) * (*xs[1]);

    }

    Matrix MatrixMultiply::backward(const std::vector<const Matrix *> &xs, const Matrix &fx, const Matrix &dedf,
                                    unsigned i) const {
        assert(i < 2);
        if (i == 0) {
            return dedf * xs[1]->transpose();
        } else {
            return xs[0]->transpose() * dedf;
        }
    }


}
