/* 
 * File:   main.cpp
 * Author: cjg
 *
 * Created on 30 settembre 2009, 12.21
 */

#include <iostream>
#include "ConfigFile.h"
#include "Communicator.h"
#include "Backend.h"

using namespace std;

int main(int argc, char** argv) {
    if(argc != 2) {
        cerr << "Usage: " << argv[0] << " /path/to/config.xml" << endl;
        return 1;
    }
    try {
        ConfigFile *cf = new ConfigFile(argv[1]);
        ConfigFile::Section communicators =
                cf->GetTopLevel()->GetSection("communicators");
        string default_communicator_name =
                cf->GetTopLevel()->GetElement("default_communicator").GetValue("value");
        ConfigFile::Element default_communicator =
                communicators.GetElement(default_communicator_name);
        Communicator *c = Communicator::Create(default_communicator);
        Backend *b = new Backend(c);
        b->Start();
        delete b;
        delete c;
        delete cf;
    } catch (string &e) {
        cerr << "Exception: " << e << endl;
    } catch(const char *e) {
        cerr << "Exception: " << e << endl;
    }
    return 0;
}

