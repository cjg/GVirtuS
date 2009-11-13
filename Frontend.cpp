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
    mpInputBuffer = new Buffer();
    mpOutputBuffer = new Buffer();
    mExitCode = cudaErrorUnknown;
    mpVar = new vector<CudaUtil::CudaVar *>();
    mAddingVar = false;
}

Frontend::~Frontend() {
    mpCommunicator->Close();
    delete mpCommunicator;
}

Frontend * Frontend::GetFrontend(bool register_var) {
    if (mspFrontend == NULL) {
        try {
            mspFrontend = new Frontend();
        } catch (const char *e) {
            cerr << "Error: cannot create Frontend ('" << e << "')" << endl;
            mspFrontend = NULL;
        }
    }
    mspFrontend->Prepare();
    if (mspFrontend->mAddingVar && !register_var) {
        for (vector<CudaUtil::CudaVar *>::iterator it = mspFrontend->mpVar->begin();
                it != mspFrontend->mpVar->end(); it++)
            mspFrontend->AddHostPointerForArguments(*it);
        mspFrontend->Execute("cudaRegisterVar");
        mspFrontend->mAddingVar = false;
        mspFrontend->Prepare();
    }
    return mspFrontend;
}

void Frontend::Execute(const char* routine, const Buffer* input_buffer) {
    if (input_buffer == NULL)
        input_buffer = mpInputBuffer;

    /* sending job */
    std::ostream &out = mpCommunicator->GetOutputStream();
    out << routine << std::endl;
    input_buffer->Dump(out);
    out.flush();

    /* receiving output */
    std::istream &in = mpCommunicator->GetInputStream();

    mpOutputBuffer->Reset();

    in.read((char *) & mExitCode, sizeof (cudaError_t));
    size_t out_buffer_size;
    in.read((char *) & out_buffer_size, sizeof (size_t));
    if (out_buffer_size > 0)
        mpOutputBuffer->Read<char>(in, out_buffer_size);
}

void Frontend::Prepare() {
    mpInputBuffer->Reset();
}

void Frontend::AddVar(CudaUtil::CudaVar* var) {
    mAddingVar = true;
    mpVar->push_back(var);
}