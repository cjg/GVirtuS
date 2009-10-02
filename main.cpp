/* 
 * File:   main.cpp
 * Author: cjg
 *
 * Created on 30 settembre 2009, 12.21
 */

#include <iostream>
#include "ConfigFile.h"
#include "AfUnixCommunicator.h"
#include "Backend.h"

using namespace std;

int main(int argc, char** argv) {
    if(argc != 2) {
        cerr << "Usage: " << argv[0] << " /path/to/config.xml" << endl;
        return 1;
    }
    ConfigFile *cf = new ConfigFile(argv[1]);
    Communicator *c = new AfUnixCommunicator("/tmp/cudactl");
    Backend *b = new Backend(c);
    b->Start();
    delete b;
    delete c;
    delete cf;
    return 0;
}

