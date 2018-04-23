//
// Created by Shuchun on 2017/12/19.
//

#include "cnn_lines.h"

#include <sstream>

using namespace std;

namespace cnn{
    string ParameterLine::to_string(const vector<string>& names)const{
        ostringstream s;
        s<<"params"<<tensor;
        return s.str();

    }

    Matrix ParameterLine::forward(const std::vector<const Matrix *> &xs) const {
        assert(x.size()==0);
        return values;
    }

    Matrix ParameterLine::backward(const vector<const Matrix*>& xs,const Matrix& fx,const Matrix& dedf, unsigned i) const{
        return Matrix();
    }

    string InLine::to_string(const std::vector<std::string> &names) const {
        ostringstream s;
        s<<"inputs"<<tensor;
        return s.str();
    }

    Matrix InLine::forward(const std::vector<const Matrix *> &xs) const {
        assert(xs.size()==0);
        return values;
    }

    Matrix InLine::backward(const std::vector<const Matrix *> &xs, const Matrix &fx, const Matrix &dedf,
                            unsigned i) const {
        return Matrix();
    }
}
