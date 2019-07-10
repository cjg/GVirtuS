#include "Process.h"

#include "util/SignalState.h"

#include <functional>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <Process.h>
#include <thread>

namespace gvirtus {

using namespace std;

Process::Process(std::shared_ptr<LD_Lib<Communicator, std::shared_ptr<gvirtus::Endpoint>>> communicator,
                 vector<string> &plugins)
    : Observable() {
  logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("Process"));
  signal(SIGCHLD, SIG_IGN);
  _communicator = communicator;
  mPlugins = plugins;
}

bool
getstring(Communicator *c, string &s) {
  s = "";
  char ch = 0;
  while (c->Read(&ch, 1) == 1) {
    if (ch == 0) {
      return true;
    }
    s += ch;
  }
  return false;
}

void
Process::Start() {
  for_each(mPlugins.begin(), mPlugins.end(), [this](const std::string &plug) {
    auto ld_path = std::filesystem::path(_PLUGINS_DIR).append("lib" + plug + "-backend.so");
    try {
      auto dl = std::make_shared<LD_Lib<Handler>>(ld_path, "create_t");
      dl->build_obj();
      _handlers.push_back(dl);
    } catch (const std::string &e) {
      LOG4CPLUS_ERROR(logger, e);
    }
  });

  // inserisci i sym dei plugin in h
  std::function<void(Communicator *)> execute = [=](Communicator *client_comm) {
    // carica i puntatori ai simboli dei moduli in mHandlers

    string routine;
    std::shared_ptr<Buffer> input_buffer = std::make_shared<Buffer>();

    while (getstring(client_comm, routine)) {
      LOG4CPLUS_DEBUG(logger, "✓ - Received routine " << routine);

      input_buffer->Reset(client_comm);

      std::shared_ptr<Handler> h = nullptr;
      for (auto &ptr_el : _handlers) {
        if (ptr_el->obj_ptr()->CanExecute(routine)) {
          h = ptr_el->obj_ptr();
          break;
        }
      }

      std::shared_ptr<Result> result;
      if (h == nullptr) {
        LOG4CPLUS_ERROR(logger, "✖ - [Process " << getpid() << "]: Requested unknown routine " << routine << ".");
        result = std::make_shared<Result>(-1, std::make_shared<Buffer>());
      } else {
        result = h->Execute(routine, input_buffer);
        // esegue la routine e salva il risultato in result
      }

      // scrive il risultato sul communicator
      //
      result->Dump(client_comm);
      if (result->GetExitCode() != 0 && routine.compare("cudaLaunch")) {
        LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "]: Requested '" << routine << "' routine.");
        LOG4CPLUS_DEBUG(logger, "✓ - - [Process " << getpid() << "]: Exit Code '" << result->GetExitCode() << "'.");
      }
    }

    Notify("process-ended");
  };

  util::SignalState sig_hand;
  sig_hand.setup_signal_state(SIGINT);

  _communicator->obj_ptr()->Serve();

  int pid = 0;
  while (true) {
    Communicator *client = const_cast<Communicator *>(_communicator->obj_ptr()->Accept());

    if (client != nullptr) {
//      if ((pid = fork()) == 0) {
        std::thread(execute, client).detach();
//        exit(0);
//      }

    } else
      _communicator->obj_ptr()->run();

    if (util::SignalState::get_signal_state(SIGINT)) {
      LOG4CPLUS_DEBUG(logger, "✓ - SIGINT received, killing server...");
      break;
    }
  }
}

Process::~Process() {
  _communicator.reset();
  _handlers.clear();
  mPlugins.clear();
}

} // namespace gvirtus
