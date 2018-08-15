#include <iostream>
#include "src/cnn.h"
#include "src/cnn_lines.h"

using namespace std;
using namespace cnn;

int main() {
//std::cout << "Hello, World!" << std::endl;

    cnn::Graph graph;
    unsigned i_x=graph.add_input(Tensor(2),"x");
    InputLine& X= reinterpret_cast<InputLine&>(*graph.lines.back());
    X(0,0)=1;
    X(1,0)=-1;
    unsigned i_y=graph.add_input(Tensor(1),"y");
    InputLine& Y= reinterpret_cast<InputLine&>(*graph.lines.back());
    Y(0,0)=1;
    unsigned i_a=graph.add_parameter(Tensor(1),"a");
    ParameterLine& a= reinterpret_cast<ParameterLine&>(*graph.lines.back());
    a(0,0)=0.1;
    unsigned i_b=graph.add_parameter(Tensor(3),"b");
    ParameterLine& b= reinterpret_cast<ParameterLine&>(*graph.lines.back());

    b(0,0)=0.3;b(1,0)=-0.02;b(2,0)=0.1;

    unsigned i_w=graph.add_parameter(Tensor(3,2),"W");
    ParameterLine& W= reinterpret_cast<ParameterLine&>(*graph.lines.back());
    W(0,0)=-0.1;W(0,1)=0.123;
    W(1,0)=0.1;W(1,1)=0.0123;
    W(2,0)=-0.15;W(2,1)=-1;

    unsigned i_v=graph.add_parameter(Tensor(1,3),"V");
    ParameterLine& V= reinterpret_cast<ParameterLine&>(*graph.lines.back());
    V(0,0)=-0.08;V(0,1)=0.22;V(0,2)=0.5;

    unsigned i_w2=graph.add_function<Square>({i_w},"W^2");
    unsigned i_v2=graph.add_function<Square>({i_v},"V^2");
    unsigned i_t1=graph.add_function<MatrixMultiply>({i_w2,i_x},"t1");
    unsigned i_t2=graph.add_function<MatrixMultiply>({i_v2,i_t1},"t2");
    unsigned i_f=graph.add_function<MatrixMultiply>({i_w,i_x},"f");
    unsigned i_g=graph.add_function<Sum>({i_f,i_b},"g");
    unsigned i_h=graph.add_function<Tanh>({i_g},"h");
    unsigned i_p=graph.add_function<MatrixMultiply>({i_v,i_h},"p");
    unsigned i_y_pred=graph.add_function<Sum>({i_p,i_a,i_t2},"y_pred");
    unsigned i_err=graph.add_function<EuclideanDistance>({i_y_pred,i_y},"err");
    graph.PrintGraph();

    cerr<<"E = "<<graph.forward()<<endl;
    graph.backward();
    cerr<<"dE/db= "<<graph.points[2]->dedf.transpose()<<endl;
    cerr<<"de/da= "<<graph.points[3]->dedf.transpose()<<endl;
    cerr<<"df/dw=\n"<<graph.points[4]->dedf<<endl;
    cerr<<"de/dv=\n"<<graph.points[5]->dedf<<endl;

    return 0;
}