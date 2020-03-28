#include "gvirtus/communicators/Result.h"

using gvirtus::communicators::Result;

Result::Result(int exit_code) {
  mExitCode = exit_code;
  mpOutputBuffer = NULL;
}

Result::Result(int exit_code, const std::shared_ptr<Buffer> output_buffer) {
  mExitCode = exit_code;
  mpOutputBuffer = (output_buffer);
}

int Result::GetExitCode() { return mExitCode; }

void Result::Dump(Communicator *c) {
  c->Write((char *)&mExitCode, sizeof(int));
  c->Write(reinterpret_cast<const char *>(&mTimeTaken), sizeof(mTimeTaken));
  if (mpOutputBuffer != NULL)
    mpOutputBuffer->Dump(c);
  else {
    size_t size = 0;
    c->Write((char *)&size, sizeof(size_t));
    c->Sync();
  }
}

void Result::TimeTaken(double time_taken) {
  mTimeTaken = time_taken;
}

double Result::TimeTaken() const {
  return mTimeTaken;
}
