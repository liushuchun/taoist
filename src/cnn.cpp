//
// Created by Shuchun on 2017/12/19.
//

#include "cnn.h"
#include "cnn_lines.h"

using namespace std;

namespace cnn {
    Line::~Line() {}

    bool Line::has_parameters() const { return false; }

    Graph::~Graph() {
        for (auto l:lines) delete l;
        for (auto p:points) delete p;
    }

    unsigned Graph::add_parameter(const Tensor &d, const std::string &name) {
        unsigned new_point_index = points.size();
        points.push_back(new Point(lines.size(), name));
        lines.push_back(new ParameterLine(d));
        lines.back()->head_point = new_point_index;
        return new_point_index;
    }


    unsigned Graph::add_input(const Tensor &d, const std::string &name) {
        unsigned new_point_index = points.size();
        points.push_back(new Point(lines.size(), name));
        lines.push_back(new InputLine(d));
        lines.back()->head_point = new_point_index;
        return new_point_index;
    }

    Matrix Graph::forward() {
        for (auto point:points) {
            const Line &in_line = *lines[point->in_line];
            vector<const Matrix *> xs(in_line.arity());
            unsigned ti = 0;
            for (unsigned tail_point_index:in_line.tail) {
                xs[ti] = &points[tail_point_index]->f;
                ++ti;
            }
            point->f = in_line.forward(xs);
            point->dedf = Zero(Tensor(point->f.rows(), point->f.cols()));
        }
        return points.back()->f;
    }


    void Graph::backward() {

        // here lets find constants to avoid doing extra job

        vector<bool> needs_derivative(points.size(), false);
        for (unsigned pi = 0; pi < points.size(); ++pi) {
            const Point &point = *points[pi];
            const Line &in_line = *lines[point.in_line];
            bool is_variable = in_line.has_parameters();
            for (auto tail_point:in_line.tail) {
                is_variable |= needs_derivative[tail_point];
            }
            needs_derivative[pi] = is_variable;
        }


        points.back()->dedf = Matrix(1, 1);
        points.back()->dedf(0, 0) = 1;

        for (int i = points.size() - 1; i >= 0; --i) {
            const Point &p = *points[i];
            const Line &in_line = *lines[p.in_line];
            vector<const Matrix *> xs(in_line.arity());
            unsigned ti = 0;
            for (unsigned tail_point_index:in_line.tail) {
                xs[ti] = &points[tail_point_index]->f;
                ++ti;
            }
            for (unsigned ti = 0; ti < in_line.tail.size(); ++ti) {
                if (needs_derivative[in_line.tail[ti]]) {
                    Point &tail_point = *points[in_line.tail[ti]];
                    tail_point.dedf += in_line.backward(xs, p.f, p.dedf, ti);
                }
            }
        }
    }


    void Graph::PrintGraph() const {
        cerr << "graph G{\n rankdir=LR;\n nodesep=.05;\n";
        unsigned nc = 0;
        for (auto point:points) {
            vector<string> var_names;
            const Line *in_line = lines[point->in_line];
            for (auto tail_point:in_line->tail) {
                var_names.push_back(points[tail_point]->variable_name());
            }
            cerr << " N" << nc << " [label=\"" << point->variable_name() << " = " << in_line->to_string(var_names)
                 << "\"];\n";
            ++nc;

        }

        for (auto line:lines) {
            for (auto ni:line->tail) {
                cerr << " N" << ni << " ->N" << line->head_point << ";\n";
            }

        }
        cerr << "}\n";

    }

}