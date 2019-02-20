#include <iostream>
#include <backend/Property.h>
#include <exception/EndpointException.h>
#include <exception/PropertyException.h>

//#include <util/JSON.h>
//int main(int argc, char **argv) {
//    gvirtus::JSON j;
//    std::filesystem::path p("/home/antonio/Desktop/GVirtuSpp/core/etc/properties.json");
//
//
//
//    try {
//        j.path(p);
//    } catch (std::fstream::failure &e) {
//        std::cerr << e.what() << std::endl;
//    }
//
//
//    return 0;








int main(int argc, char **argv) {
    gvirtus::Property p;
    std::vector<gvirtus::Endpoint> ends;
    std::vector<std::string> plugins;


    try {
        gvirtus::Endpoint endpoint;
        endpoint.protocol("http").address("192.168.1.1").port("9998");
        ends.emplace_back(endpoint);
    }
    catch (gvirtus::EndpointException &exception) {
        std::cout << "WHAT: " << exception.what() << std::endl;
        std::cout << "FILE: " << exception.get_file() << std::endl;
        std::cout << "LINE: " << exception.get_line() << std::endl;
        std::cout << "FUNCTION: " << exception.get_func() << std::endl;
        std::cout << "INFO: " << exception.get_info() << std::endl;
    }

    p.endpoints(&ends);
    plugins.emplace_back("we");
    try {
        p.plugins(&plugins);
    } catch (gvirtus::PropertyException &exception1) {
        std::cout << "WHAT: " << exception1.what() << std::endl;
        std::cout << "FILE: " << exception1.get_file() << std::endl;
        std::cout << "LINE: " << exception1.get_line() << std::endl;
        std::cout << "FUNCTION: " << exception1.get_func() << std::endl;
        std::cout << "INFO: " << exception1.get_info() << std::endl;
    }

    for (const auto &end : p.endpoints()) {
        std::cout << "Protocollo: " << end.protocol() << std::endl;
        std::cout << "Indirizzo: " << end.address() << std::endl;
        std::cout << "Porta: " << end.port() << std::endl;
    }

    for (const auto &el : p.plugins()) {
        std::cout << "Plugin: " << el << std::endl;
    }

    return 0;
}