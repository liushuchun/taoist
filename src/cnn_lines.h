#ifndef CNN_LAYERS_H_
#define CNN_LAYERS_H_

#include "cnn.h"

using namespace std;
namespace cnn {
    struct ParameterLine : public Line {
        ParameterLine(const Tensor &t) : tensor(t), values(Random(t)) {}

        bool has_parameters() const override;

        std::string to_string(const std::vector<std::string> &arg_names) const override;

        Matrix forward(const std::vector<const Matrix *> &xs) const override;

        Matrix backward(const std::vector<const Matrix *> &xs, const Matrix &fx, const Matrix &dedf,
                        unsigned i) const override;

        inline real &operator()(int i, int j) { return values(i, j); }

        inline const real &operator()(int i, int j) const { return values(i, j); }

        Tensor tensor;
        Matrix values;
    };

    struct InputLine : public Line {
        InputLine(const Tensor &t) : tensor(t), values(Zero(t)) {}

        std::string to_string(const std::vector<std::string> &names) const override;

        Matrix forward(const std::vector<const Matrix *> &xs) const override;

        Matrix backward(const std::vector<const Matrix *> &xs, const Matrix &fx, const Matrix &dedf,
                        unsigned i) const override;

        inline real &operator()(int i, int j) { return values(i, j); }

        inline const real &operator()(int i, int j) const { return values(i, j); }

        Tensor tensor;
        Matrix values;
    };


    struct MatrixMultiply : public Line {
        std::string to_string(const std::vector<std::string> &args) const override;

        Matrix forward(const std::vector<const Matrix *> &xs) const override;


        Matrix backward(const std::vector<const Matrix *> &xs, const Matrix &fx, const Matrix &dedf,
                        unsigned i) const override;


    };

    struct LogSoftmax : public Line {
        //z=sum_j/exp(x_i)_j y_i=(x_1)_1 i\log z

        string to_string(const vector<string> &args) const {
            ostringstream s;
            s << "Log Softmax(" << args[0] << ")";
            return s.str();
        }

        Matrix foward(const vector<const Matrix *> &xs) const {
            assert(xs.size() == 1);
            const Matrix &x = *xs.front();
            const unsigned rows = x.rows();
            assert(x.cols() == 1);
            Matrix fx(rows, 1);
            //TODO switch to logsum and z=-inf
            real z = 0;
            for (unsigned i = 0; i < rows; i++) {
                z += exp(x(i, 0));
            }
            real logz = log(z);
            for (unsigned i = 0; i < rows; ++i) {
                fx(i, 0) = x(i, 0) - logz;

            }
            return fx;
        }

        Matrix
        backward(const vector<const Matrix *> &xs, const Matrix &fx, const Matrix &dedf, unsigned i) const override {
            assert(i == 0);
            const Matrix &x = *xs.front();
            const unsigned rows = x.rows();
            Matrix dedx(rows, 1);
            double z = 0;
            for (unsigned i = 0; i < rows; ++i) {
                z += dedf(i, 0);
            }
            for (unsigned i = 0; i < rows; ++i) {
                dedx(i, 0) = dedf(i, 0) - exp(fx(i, 0)) * z;
            }
            return dedx;
        }
    };

    struct SelectItem : public Line {
        //x1 is a vect
        //x2 is a scalar index stored in (0,0) ,y=(x1)_{x2}
        string to_string(const vector<string> &args) const {
            ostringstream s;
            s << "pick(" << args[0] << "_" << args[1] << ")";
            return s.str();
        }

        Matrix forward(const vector<const Matrix *> &xs) const {
            assert(xs.size() == 2);
            const Matrix &x = *xs.front();
            assert(x.cols() == 1);
            const Matrix &mindex = *xs.back();
            assert(mindex.rows() == 1);
            assert(mindex.cols() == 1);
            const unsigned index = static_cast<unsigned>(mindex(0, 0));
            assert(index < x.rows());
            Matrix fx(1, 1);
            fx(0, 0) = x(index, 0);
            return fx;
        }

        //derivative is o in all dimensions except 1 for the selected item
        Matrix
        backward(const vector<const Matrix *> &xs, const Matrix &fx, const Matrix &dedf, unsigned i) const override {
            assert(i == 0);
            assert(dedf.rows() == 1);
            assert(dedf.cols() == 1);
            const Matrix &x = *xs.front();
            const Matrix& mindex = *xs.back();

            Matrix dedx1 = Matrix::Zero(x.rows(), 1);
            dedx1(int(mindex(0, 0)), 0) = dedf(0, 0);
            return dedx1;
        }


    };

    struct Sum : public Line {
        string to_string(const vector<string> &names) const {
            ostringstream s;
            s << names[0];
            for (auto i = 1; i < tail.size(); ++i) {
                s << " + " << names[1];
            }
            return s.str();
        }

        Matrix forward(const vector<const Matrix *> &xs) const {
            assert(xs.size() > 1);
            Matrix res = *xs[0];
            for (unsigned i = 1; i < xs.size(); ++i) {
                res += *xs[i];
            }
            return res;
        }

        Matrix
        backward(const vector<const Matrix *> &xs, const Matrix &fx, const Matrix &dedf, unsigned i) const override {
            return dedf;
        }
    };

    struct EuclideanDistance : public Line {
        //y=||x1-x2||^2
        string to_string(const vector<string> &names) const {
            ostringstream s;
            s << "|| " << names[0] << " - " << names[1] << " ||^2";
            return s.str();
        }

        Matrix forward(const vector<const Matrix *> &xs) const {
            assert(xs.size() == 2);
            Matrix res(1, 1);
            res(0, 0) = (*xs[0] - *xs[1]).squaredNorm();
            return res;
        }

        Matrix
        backward(const vector<const Matrix *> &xs, const Matrix &fx, const Matrix &dedf, unsigned i) const override {
            assert(i < 2);
            real scale = dedf(0, 0) * 2;
            if (i == 1) scale = -scale;
            return scale * (*xs[0] - *xs[1]);
        }
    };

    struct LogisticSigmoid : public Line {
        //y=sigmoid x_1
        string to_string(const vector<string> &names) const {
            ostringstream s;
            s << "sigma(" + names[0] << ")";
            return s.str();
        }


        Matrix forward(const vector<const Matrix *> &xs) const {
            assert(xs.size() == 1);
            const Matrix &x = *xs.front();
            const unsigned rows = x.rows();
            const unsigned cols = x.cols();
            Matrix fx(rows, cols);
            for (unsigned i = 0; i < rows; ++i) {
                for (unsigned j = 0; j < cols; ++j) {
                    fx(i, j) = 1. / (1. + exp(-x(i, j)));
                }
            }
            return fx;
        }

        Matrix
        backward(const vector<const Matrix *> &xs, const Matrix &fx, const Matrix &dedf, unsigned i) const override {
            assert(i == 0);
            const Matrix &x = *xs.front();
            const unsigned rows = x.rows();
            const unsigned cols = x.cols();
            Matrix dfdx(rows, cols);
            for (unsigned i = 0; i < rows; ++i) {
                for (unsigned j = 0; j < cols; ++j) {
                    dfdx(i, j) = (1. - fx(i, j)) * fx(i, j);

                }
            }
            return dfdx.cwiseProduct(dedf);
        }


    };


    struct Tanh : public Line {
        //y=tanh x_1
        string to_string(const vector<string> &names) const {
            ostringstream s;
            s << "tan(" << names[0] << ")";
            return s.str();
        }

        Matrix forward(const vector<const Matrix *> &xs) const {
            assert(xs.size() == 1);
            const Matrix &x = *xs.front();
            const unsigned rows = x.rows();
            const unsigned cols = x.cols();
            Matrix fx(rows, cols);
            for (unsigned i = 0; i < rows; ++i) {
                for (unsigned j = 0; j < cols; ++j) {
                    fx(i, j) = tanh(x(i, j));
                }
            }
            return fx;
        }

        Matrix
        backward(const vector<const Matrix *> &xs, const Matrix &fx, const Matrix &dedf, unsigned i) const override {
            assert(i == 0);
            const Matrix &x = *xs.front();
            const unsigned rows = x.rows();
            const unsigned cols = x.cols();
            Matrix dfdx(rows, cols);
            for (unsigned i = 0; i < rows; ++i) {
                for (unsigned j = 0; j < cols; ++j) {
                    dfdx(i, j) = 1. - fx(i, j) * fx(i, j);
                }
            }
            return dfdx.cwiseProduct(dedf);
        }


    };

    struct Square : public Line {
        //Y=X_1*X_1
        string to_string(const vector<string> &names) const {
            ostringstream s;
            s << "square(" + names[0] << ")";
            return s.str();
        }

        Matrix forward(const vector<const Matrix *> &xs) const {
            assert(xs.size() == 1);
            const Matrix &x = *xs.front();
            return x.cwiseProduct(x);
        }

        Matrix
        backward(const vector<const Matrix *> &xs, const Matrix &fx, const Matrix &dedf, unsigned i) const override {
            assert(i == 0);
            const Matrix val = *xs.front();
            return dedf.cwiseProduct(val) * 2;
        }

    };


}

#endif