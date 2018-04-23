#ifndef CNN_LAYERS_H_
#define CNN_LAYERS_H_

#include "cnn.h"

namespace  cnn{
    struct ParameterLine:public Line{
        ParameterLine(const Tensor& d):tensor(t),values(Random(t)){}
        std::string to_string(const std::vector<std::string>& arg_names)const override;
        Matrix forward(const std::vector<const Matrix*>& xs) const override ;
        Matrix backward(const std::vector<const Matrix*>& xs,const Matrix& fx,const Matrix& dedf,unsigned  i) const override;
        inline real& operator()(int i,int j)const{return values(i,j);}
        Tensor tensor;
        Matrix values;
    };

    struct InLine:public Line{
        InLine(const Tensor& t):tensor(t),values(Zero(t)){}
        std::string to_string(const std::vector<std::string>& names)const override;
        Matrix forward(const std::vector<const Matrix*>& xs) const override;
        Matrix backward(const std::vector<const Matrix*>& xs,const Matrix& fx,const Matrix& dedf, unsigned i) const override;
        inline real& operator()(int i,int j){return values(i,j);}
        linline const real& operator()(int i,int j)const{return values(i,j);}
        Tensor tensor;
        Matrix values;
    };

    using namespace std;

    struct MatrixMultiply:public Line{
        std::string to_string(const std::vector<std::string>& names) const{
            ostringstream s;
            s<<names[0]<<" * "<<names[1];
            return s.str();
        }

        Matrix forward(const std::vector<const Matrix*>& xs)const{
            assert(xs.size()==2);
            return (*xs[0])*(*xs[1]);
        }

        Matrix backward(const std::vector<const Matrix*>& xs) const{
            assert(xs.size()==2);
            return (*xs[0])*(*xs[1]);
        }


        Matrix backward(const std::vector<const Matrix*>& xs,const Matrix& fx,const Matrix& dedf, unsigned i) const override {
            assert(i<2);
            if(i==0){
                return dedf*xs[1]->transpose();
            }else{
                return xs[0]->transpose()*dedf;
            }
        }
    };

    struct Sum:public Line{
        string to_string(const vector<string>& names) const{
            ostringstream s;
            s<<names[0];
            for(unsigned i=1;i<tail.size(),++i){
                s<<" + "<<names[1];
            }
            return s.str();
        }

        Matrix forward(const vector<const Matrix*>& xs) const{
            assert(x.size()>1);
            Matrix res=*xs[0];
            for(unsigned i=1;i<x.size();++i){
                res+=*xs[i];
            }
            return res;
        }

        Matrix backward(const vector<const Matrix*>& xs,const Matrix& fx,const Matrix& dedf,unsigned i)const override{
            return dedf;
        }
    };

    struct EuclideanDistance:public Line{
        //y=||x1-x2||^2
        string to_string(const vector<string>& names)const{
            ostringstream s;
            s<<"|| "<<names[0]<<" - "<<names[1]<<" ||^2";
            return s.str();
        }

        Matrix forward(const vector<const Matrix*>& xs)const{
            assert(xs.size()==2);
            Matrix res(1,1);
            res(0,0)=(*xs[0]-*xs[1]).squaredNorm();
            return res;
        }

        Matrix backward(const vector<const Matrix*>& xs,const Matrix& fx,const Matrix& dedf, unsigned i) const override{
            assert(i<2);
            real scale=dedf(0,0)*2;
            if(i==1) scale=-scale;
            return scale*(*xs[0]-*xs[1]);
        }
    };

    struct LogisticSigmoid:public Line{
        //y=sigmoid x_1
        string to_string(const vector<string>& names)const{
            ostringstream s;
            s<< "sigma("+names[0]<<")";
            return s.str();
        }


        Matrix forward(const vector<const Matrix*>& xs) const{
            assert(xs.size()==1);
            const Matrix& x=*xs.front();
            const unsigned rows=x.rows();
            const unsigned cols=x.cols();
            Matrix fx(rows,cols);
            for(unsigned i=0;i<rows;++i){
                for(unsigned j=0;j<cols;++j){
                    fx(i,j)=1./(1.+exp(-x(i,j)));
                }
            }
            return fx;
        }

        Matrix backward(const vector<const Matrix*>& xs,const Matrix& fx,const Matrix& dedf, unsigned i)const override{
            assert(i==0);
            const Matrix& x=*xs.front();
            const unsigned  rows=x.rows();
            const unsigned cols=x.cols();
            Matrix dfdx(rows,cols);
            for(unsigned i=0;i<rows;++i){
                for(unsigned j=0;j<cols;++j){
                    dfdx(i,j)=(1.-fx(i,j))*fx(i,j);

                }
            }
            return dfdx.cwiseProduct(dedf);
        }


    };


    struct Tanh:public line{
        //y=tanh x_1
        string to_string(const vector<string>& names) const{
            ostringstream s;
            s<<"tan("<<names[0]<<")";
            return s.str();
        }

        Matrix forward(const vector<const Matrix*>& xs) const{
            assert(xs.size()==1);
            const Matrix& x=&xs.front();
            const unsigned rows=x.rows();
            const unsigned cols=x.cols();
            Matrix fx(rows,cols);
            for(unsigned i=0;i<rows;++i){
                for(unsigned j=0;j<cols;++j){
                    fx(i,j)=tanh(x(i,j));
                }
            }
            return fx;
        }

        Matrix backward(const vector<const Matrix*>& xs,const Matrix& fx,const Matrix& dedf, unsigned i) const override{
            assert(i==0);
            const Matrix& x=*xs.front();
            const unsigned rows=x.rows();
            const unsigned cols=x.cols();
            Matrix dfdx(rows,cols);
            for(unsigned i=0;i<rows;++i){
                for(unsigned j=0;j<cols;++j){
                    dfdx(i,j)=1.-fx(i,j)*fx(i,j);
                }
            }
            return dfdx.cwiseProduct(dedf);
        }


    };

    struct Square:public Line{
        //Y=X_1*X_1
        string to_string(const vector<string>& names) const{
            ostringstream s;
            s<<"square("+names[0]<<")";
            return s.str();
        }

        Matrix forward(const vector<const Matrix*>& xs) const{
            assert(xs.size()==1);
            const Matrix& x=*xs.front();
            return x.cwiseProduct(x);
        }

        Matrix backward(const vectorM<const Matrix*>& xs,const Matrix& fx,const Matrix& dedf,unsigned i) const override{
            assert(i==0);
            return dedf,cwiseProduct(*xs.front())*2;
        }

    };



}

#endif