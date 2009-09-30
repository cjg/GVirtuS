/* 
 * File:   Frontend.cpp
 * Author: cjg
 * 
 * Created on 30 settembre 2009, 12.57
 */

#include "AfUnixCommunicator.h"
#include "Frontend.h"

Frontend::Frontend() {
    // TODO: Communicator to use should be obtained from config file
    mpCommunicator = new AfUnixCommunicator("/tmp/cudactl");
    mpCommunicator->Connect();
}


Frontend::~Frontend() {
    mpCommunicator->Close();
}

Frontend & Frontend::GetFrontend() {
    if(mspFrontend == NULL)
        mspFrontend = new Frontend();
    return *mspFrontend;
}

int Frontend::Execute(const char* routine, const char* in_buffer,
        int in_buffer_size, char** out_buffer, int* out_buffer_size) {

    /* sending job */
    std::ostream &out = mpCommunicator->GetOutputStream();
    out << routine << std::endl;
    out.write(in_buffer, in_buffer_size);
    out.flush();

    /* receiving output */
    std::istream &in = mpCommunicator->GetInputStream();
    int result;
    in.read((char *) &result, sizeof(int));
    in.read((char *) out_buffer_size, sizeof(int));
    if(*out_buffer_size > 0) {
        *out_buffer = new char[*out_buffer_size];
        in.read(*out_buffer, *out_buffer_size);
    } else
        *out_buffer = NULL;

    return result;
}

