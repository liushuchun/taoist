//
// Created by Shuchun on 2018/8/16.
//

#include "cnn/cnn-lines.h"
#include "cnn/cnn.h"

#inlcude <iostream>
using namespace std;
using namespace cnn;


int main(){
    sranddev();

    cnn::Graph graph;
    unsigned i_x=graph.add_input(Tensor(2),"x");
}