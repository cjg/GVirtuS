#include "HTTPAdapter.h"
#include "Buffer.h"

namespace gvirtus {

HTTPAdapter::HTTPAdapter(const std::string &communicator) {

}

HTTPAdapter::HTTPAdapter(std::string &hostname, std::string &port) {
  logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("HTTP Communicator"));
  _hostname = hostname;
  _port = port;
  _threads = std::vector<std::unique_ptr<std::thread>>(std::thread::hardware_concurrency());
}

void HTTPAdapter::Serve() {
  std::transform(_threads.begin(), _threads.end(), _threads.begin(), [this](std::unique_ptr<std::thread> &t) {
    return std::make_unique<std::thread>([this]() {

      uWS::App().get("/hello", [](auto *res, auto *req) {
        res->end("Hello world!");
      }).ws<std::string>("/", {
          /* Settings */
          .compression = uWS::SHARED_COMPRESSOR,
          .maxPayloadLength = 16 * 1024,
          .idleTimeout = 120,

          /* Handlers */
          .open = [this](auto *ws, auto *req) {
            LOG4CPLUS_DEBUG(logger, "✓ - WSS Connected " << _port);
          },
          .message = [this](uWS::WebSocket<false, true> *ws, std::string_view message, uWS::OpCode opCode) {
            //TODO:      ws->send(message, opCode, true);
            LOG4CPLUS_DEBUG(logger, "✓ - Thread " << std::this_thread::get_id() << "  message: " << message);
          }
      }).listen(std::stoi(_port), [this](us_listen_socket *token) {
        if (token) {
          LOG4CPLUS_DEBUG(logger, "✓ - Thread " << std::this_thread::get_id() << " listening on port " << _port);
        }
      }).run();
    });
  });
}

const Communicator *const HTTPAdapter::Accept() const {
  return nullptr;
}

void HTTPAdapter::run() {

}

void HTTPAdapter::Connect() {
  //FIXME: non funziona
  std::string url = _hostname + ":" + _port + "/" + "hello";

  try {
    uWS::App().connect(url, [](uWS::HttpResponse<false> *res, uWS::HttpRequest *req) {
      std::cout << req->getMethod() << std::endl;
    });
  }
  catch (std::runtime_error &e) {
    std::cout << e.what() << __FILE__ << std::endl;
  }
}

HTTPAdapter::~HTTPAdapter() {
  _threads.clear();
}

extern "C" std::shared_ptr<HTTPAdapter>
create_communicator(std::string &arg) {
  auto colon = arg.rfind(":");
  auto hostname = arg.substr(0, colon);
  auto port = arg.substr(colon + 1);

  return std::make_shared<HTTPAdapter>(hostname, port);
}

}