#ifndef CNN_CNN_H_
#define CNN_CNN_H_

#include <string>
#include <vector>
#include <iostream>
#include <initializer_list>
#include <Eigen/Eigen>

// computation graph where points represent forward & backward intermediate
//values,and lines represent fucntions of multipe values.to represent the
//fact that a fucntion may have multiple arguments,edeges have a single head
//and 0,1,2,or more tails.(constants,inputs,and parameters are represented as
//function of 0 parameters.


namespace cnn{
    typedef Eigen::MatrixXd Matrix;
    typedef double real;


    //TODO pull fx and dedf out of the Point object and have them as local tables in forward/backward algorithms
    //TODO figure out what paths are constants and don't propagate errors along them


    struct Tensor{
        Tensor():rows(1),cols(1){}
        explicit Tensor(unsigned m): rows(m),cols(1){}
        Tensor(unsigned m , unsigned n):rows(m),cols(n){}
        unsigned short rows;
        unsigned short cols;
        Tensor transpose() const{return Tensor(cols,rows);}
    };

    inline Tensor operator*(const Tensor& a,const Tensor& b){
        assert(a.cols==b.rows);
        return Tensor(a.rows,b.cols);
    }

    inline std::ostream& operator<<(std::ostream& os,const Tensor& d){
        return os<<'('<<d.rows<<','<<d.cols<<')';
    }

    inline Matrix Zero(const Tensor& d){return Matrix::Zero(d.rows,d.cols);}

    inline Matrix Random(const Tensor& d){return Matrix::Random(d.rows,d.cols);}


    struct Line;

    struct Point;


    struct Graph{
        ~Graph();
        //construct a graph
        unsigned  add_parameter(const Tensor& d,const std::string& name = "");
        unsigned  add_input(const Tensor& d,const std::string& name="");

        template <class Function> inline unsigned  add_function(const std::initializer_list<unsigned >& arguments,const std::string& name="");

        //perform computations
        Matrix forward();
        void backward();

        //debuging
        void PrintGraph() const;

        //data

        std::vector<Line*> lines;

        std::vector<Point*> points; //stored in topological orders

    };


    // represents an SSA variable
    // *in_line is the index of the function that computes the variable
    //  *out_lines are the list of functions that use this variable
    // *f is the computed value of the variable(todo:remove this)
    // *dEdf is the derivative of the output with respect to the function

    struct Point{
        Point(unsigned in_line_index,const std::string& name):
                in_line(in_line_index),var_name(name){}

        //depending
        unsigned in_line;
        std::vector<unsigned > out_lines;


        const std::string& variable_name()const{return var_name;}

        std::string var_name;

        //computation
        // todo remove thes from here, they shold be local to the forward/backward
        //algorithms
        Matrix f; //(f(x_1,...,x_n)
        Matrix dedf; //dE/df


    };

    struct Line{
        virtual  ~Line();
        //debugging
        virtual std::string to_string(const std::vector<std::string>& var_names) const=0;


        //computation
        virtual Matrix forward(const std::vector<const Matrix*>& xs) const=0;
        // computes the derivative of E with respect to the ith argument to f,that is,xs[i]
        virtual Matrix backward(const std::vector<const Matrix*>& xs,const Matrix& fx,const Matrix& dEdf, unsigned i) const=0;

        //number of arguments to the function
        inline unsigned arity() const{return tail.size();}

        //structure
        unsigned head_point; //index of point to contain result of f
        std::vector<unsigned> tail;
    };
    //add computing graph
    template <class Function>
    inline unsigned Graph::add_function(const std::initializer_list<unsigned>& arguments,const std::string& name){
        unsigned new_point_index=points.size();
        unsigned new_line_index=lines.size();
        points.push_back(new Point(new_line_index,name));
        Line* new_line=new Function;
        lines.push_back(new_line);
        new_line->head_point=new_point_index;
        for(auto ni:arguments){
            new_line->tail.push_back(ni);
            points[ni]->out_lines.push_back(new_line_index);
        }
        return new_point_index;
    }




}


#endif



