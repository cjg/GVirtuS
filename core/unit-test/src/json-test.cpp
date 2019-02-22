#include <iostream>
#include <util/JSON.h>
#include <backend/Property.h>

int main(int argc, char **argv) {
    gvirtus::JSON<gvirtus::Property> j;
    std::filesystem::path p("/home/antonio/Desktop/GVirtuSpp/etc/properties.json");


    try {
        j.path(p);
        auto prop = j.parser();
            std::cout << prop.plugins().size() << std::endl;

    } catch (std::fstream::failure &e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}