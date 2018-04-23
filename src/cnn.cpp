//
// Created by Shuchun on 2017/12/19.
//

#include "cnn.h"
#include "cnn_lines.h"

using namespace std;

namespace cnn{
    Line::~Line(){}

    Graph::~Graph() {
        for (auto l:lines) delete l;
        for (auto p:points) delete n;
    }

    unsigned Graph::add_parameter(const Tensor &d, const std::string &name) {
        unsigned new_point_index=points.size();
        points.push_back(new Point(lines.size(),name));
        lines.push_back(new ParameterLine(d));
        lines.back()->head_point=new_point_index;
        return new_point_index;
    }


    unsigned Graph::add_input(const Tensor &d, const std::string &name) {
        unsigned new_point_index=points.size();
        points.push_back(new Point(lines.size(),name));
        lines.push_back(new InLine(d));
        lines.back()->head_point=new_point_index;
        return new_point_index;
    }

    Matrix Graph::forward() {
        for(auto point:points){
            const Line& in_line=*lines[point->in_line];
            vector<const Matrix*> xs(in_line.arity());
            unsigned  ti=0;
            for(unsigned tail_point_index:in_line.tail){
                xs[ti]=&points[tail_point_index]->f;
                ++ti;
            }
            point->f=in_line.forward(xs);
            point->dedf=Zero(Tensor(point->f.rows(),point->f.cols()));
        }
        return points.back()->f;
    }


    void Graph::backward() {
        points.back()->dedf=Matrix(1,1);
        points.back()->dedf(0,0)=1;

        for(int i=points.size()-1;i>=0;--i){
            const Point& p=*points[i];
            const Line& in_line=*lines[p.in_line];
            vector<const Matrix*> xs(in_line.arity());
            unsigned ti=0;
            for(unsigned tail_point_index:in_line.tail){
                xs[ti]=&points[tail_point_index]->f;
                ++ti;
            }
            for(unsigned ti=0;ti<in_line.tail.size();++ti){
                Point& tail_point=*nodes[in_line.tail[ti]];
                tail_point.dedf+=in_line.backward(xs,point.f,point.dedf,ti);
            }
        }
    }
}