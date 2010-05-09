/* 
 * File:   Communicator.cpp
 * Author: cjg
 * 
 * Created on 30 settembre 2009, 11.56
 */

#include <cstring>
#include "AfUnixCommunicator.h"
#include "TcpCommunicator.h"
#include "VmciCommunicator.h"
#include "VMSocketCommunicator.h"
#include "Communicator.h"

Communicator * Communicator::Create(ConfigFile::Element & config) {
    const char *type = config.GetValue("type").c_str();
    if (strcasecmp(type, "AfUnix") == 0) {
        mode_t mode = 0660;
        bool use_shm = false;
        if(config.HasKey("mode"))
            mode = config.GetShortValueFromOctal("mode");
        if(config.HasKey("use_shm"))
            use_shm = config.GetBoolValue("use_shm");
        return new AfUnixCommunicator(config.GetValue("path"), mode, use_shm);
    } else if (strcasecmp(type, "Tcp") == 0)
        return new TcpCommunicator(
                config.GetValue("hostname").c_str(),
                config.GetShortValue("port"));
#ifdef HAVE_VMCI
    else if (strcasecmp(type, "Vmci") == 0)
        return new VmciCommunicator(config.GetShortValue("port"),
                config.GetShortValue("cid"));
#endif
    else if (strcasecmp(type, "VMSocket") == 0)
        if(config.HasKey("shm"))
            return new VMSocketCommunicator(config.GetValue("device"),
                    config.GetValue("shm"));
        else
            return new VMSocketCommunicator(config.GetValue("device"));
    else
        throw "Not a valid type!";
    return NULL;
}

Communicator::~Communicator() {
}
