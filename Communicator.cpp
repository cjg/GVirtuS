/* 
 * File:   Communicator.cpp
 * Author: cjg
 * 
 * Created on 30 settembre 2009, 11.56
 */

#include <cstring>
#include "Communicator.h"
#include "AfUnixCommunicator.h"

Communicator * Communicator::Create(ConfigFile::Element & config) {
    if(strcasecmp(config.GetValue("type").c_str(), "AfUnix") == 0) {
        return new AfUnixCommunicator(config.GetValue("path"));
    } else
        throw "Not a valid type!";
}

Communicator::~Communicator() {
}

