/* 
 * File:   Frontend.cpp
 * Author: cjg
 * 
 * Created on 30 settembre 2009, 12.57
 */

#include <iostream>
#include <string>
#include "ConfigFile.h"
#include "Frontend.h"

using namespace std;

Frontend *Frontend::mspFrontend = NULL;

Frontend::Frontend() {
    ConfigFile *cf = new ConfigFile(_CONFIG_FILE);
    ConfigFile::Section communicators =
            cf->GetTopLevel()->GetSection("communicators");
    string default_communicator_name =
            cf->GetTopLevel()->GetElement("default_communicator").GetValue("value");
    ConfigFile::Element default_communicator =
            communicators.GetElement(default_communicator_name);
    mpCommunicator = Communicator::Create(default_communicator);
    mpCommunicator->Connect();
}

Frontend::~Frontend() {
    mpCommunicator->Close();
    delete mpCommunicator;
}

Frontend & Frontend::GetFrontend() {
    if (mspFrontend == NULL) {
        try {
            mspFrontend = new Frontend();
        } catch (const char *e) {
            cerr << "Error: cannot create Frontend ('" << e << "')" << endl;
            mspFrontend = NULL;
        }
    }
    return *mspFrontend;
}

cudaError_t Frontend::Execute(const char* routine, const char* in_buffer,
        size_t in_buffer_size, char** out_buffer, size_t* out_buffer_size) {

    /* sending job */
    std::ostream &out = mpCommunicator->GetOutputStream();
    out << routine << std::endl;
    out.write(in_buffer, in_buffer_size);
    out.flush();

    /* receiving output */
    std::istream &in = mpCommunicator->GetInputStream();

    cudaError_t result;
    in.read((char *) & result, sizeof (int));
    in.read((char *) out_buffer_size, sizeof (size_t));
    if (*out_buffer_size > 0) {
        *out_buffer = new char[*out_buffer_size];
        in.read(*out_buffer, *out_buffer_size);
    } else
        *out_buffer = NULL;

    return result;
}

