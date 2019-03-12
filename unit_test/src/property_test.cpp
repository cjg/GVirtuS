#include <backend/Property.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using testing::Eq;

namespace {
    class Property_test : public testing::Test {
    public:
        gvirtus::Property obj;

        Property_test() {
            obj;
        }
    };
}

TEST_F(Property_test, endpoints_test) {
    gvirtus::Endpoint end;
    end.protocol("tcp").address("127.0.0.1").port("9999");
    ASSERT_EQ(gvirtus::Endpoint("tcp", "127.0.0.1", "9999").to_string(), end.to_string());

    std::vector<gvirtus::Endpoint> ends;
    ends.emplace_back(end);
    obj.endpoints(&ends);
    ASSERT_EQ(ends, obj.endpoints());
}

TEST_F(Property_test, plugins_test) {
    std::vector<std::string> plug;
    plug.emplace_back("cudart");

    obj.plugins(&plug);
    ASSERT_EQ(plug, obj.plugins());
}

