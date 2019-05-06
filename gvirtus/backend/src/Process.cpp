#include "Process.h"
#include <dlfcn.h>
#include <functional>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>

namespace gvirtus {

  using namespace std;

  static GetHandler_t
  LoadModule(const char *name) {
    log4cplus::Logger logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("LoadModule"));
    char path[4096];
    if (*name == '/')
      strcpy(path, name);
    else
      sprintf(path, _PLUGINS_DIR "/lib%s-backend.so", name);

    void *lib = dlopen(path, RTLD_LAZY);
    if (lib == NULL) {
      cerr << "Error loading " << path << ": " << dlerror() << endl;
      return NULL;
    }

    HandlerInit_t init = (HandlerInit_t)((pointer_t)dlsym(lib, "HandlerInit"));
    if (init == NULL) {
      dlclose(lib);
      cerr << "Error loading " << name << ": HandlerInit function not found." << endl;
      return NULL;
    }

    if (init() != 0) {
      dlclose(lib);
      cerr << "Error loading " << name << ": HandlerInit failed." << endl;
      return NULL;
    }

    GetHandler_t sym = (GetHandler_t)((pointer_t)dlsym(lib, "GetHandler"));
    if (sym == NULL) {
      dlclose(lib);
      cerr << "Error loading " << name << ": " << dlerror() << endl;
      return NULL;
    }

    LOG4CPLUS_DEBUG(logger, "✓ - Loaded module '" << name << "'.");

    return sym;
  }

  Process::Process(std::unique_ptr<comm::Communicator> communicator, vector<string> &plugins) : Observable() {
    logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("Process"));
    signal(SIGCHLD, SIG_IGN);
    _server_communicator = std::move(communicator);
    mPlugins = plugins;
  }

  bool
  getstring(comm::Communicator *c, string &s) {
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
    GetHandler_t h;
    for (vector<string>::iterator i = mPlugins.begin(); i != mPlugins.end(); i++) {
      if ((h = LoadModule((*i).c_str())) != NULL)
        mHandlers.push_back(h());
    }

    std::function<void(comm::Communicator *)> execute = [=](comm::Communicator *client_comm) {
      LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "]: Started.");

      // carica i puntatori ai simboli dei moduli in mHandlers

      string routine;
      // TODO: MAKE SMART POINTER
      Buffer *input_buffer = new Buffer();

      while (getstring(client_comm, routine)) {
        LOG4CPLUS_DEBUG(logger, "✓ - Received routine " << routine);

        input_buffer->Reset(client_comm);

        Handler *h = NULL;
        for (vector<Handler *>::iterator i = mHandlers.begin(); i != mHandlers.end(); i++) {
          if ((*i)->CanExecute(routine)) {
            h = *i;
            break;
          }
        }

        // leggo il nome della routine, vedo quale handler gestisce la routine con CanExecute
        // h punterà all'handler che gestisce la routine

        Result *result;
        if (h == NULL) {
          LOG4CPLUS_ERROR(logger, "✖ - [Process " << getpid() << "]: Requested unknown routine " << routine << ".");
          result = new Result(-1, new Buffer());
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
        delete result;
      }

      Notify("process-ended");

      LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "]: Finished.");
    };

    _server_communicator->Serve();

    int pid = 0;
    while (true) {
      comm::Communicator *client = const_cast<comm::Communicator *>(_server_communicator->Accept());
      LOG4CPLUS_DEBUG(logger, "✓ - Connection accepted");

      if ((pid = fork()) == 0) {
        execute(client);
        exit(0);
      }
    }
  }
} // namespace gvirtus
