#include "Result.h"

Result::Result(int exit_code) {
    mExitCode = exit_code;
    mpOutputBuffer = NULL;
}

Result::Result(int exit_code, const std::shared_ptr<Buffer> output_buffer) {
    mExitCode = exit_code;
    mpOutputBuffer = (output_buffer);
}

int Result::GetExitCode() {
    return mExitCode;
}

void
Result::Dump(gvirtus::Communicator *c) {
  c->Write((char *)&mExitCode, sizeof(int));
  if (mpOutputBuffer != NULL)
    mpOutputBuffer->Dump(c);
  else {
    size_t size = 0;
    c->Write((char *)&size, sizeof(size_t));
    c->Sync();
  }
}
