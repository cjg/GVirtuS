#include <communicator/endpoint/Endpoint.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using testing::Eq;

namespace {
    class Endpoint_test : public testing::Test {
    public:
        gvirtus::Endpoint obj;

        Endpoint_test() {
            obj;
        }
    };
}

TEST_F(Endpoint_test, constructor){
    obj.protocol("tcp").address("127.0.0.1").port("9999");
    auto obj2 = gvirtus::Endpoint("tcp", "127.0.0.1", "9999");
    ASSERT_EQ(obj2, obj);
}

TEST_F(Endpoint_test, port_test) {
    auto obj2 = obj.port("9999");
    ASSERT_EQ(9999, obj.port());
    ASSERT_EQ(obj2, obj);

}

TEST_F(Endpoint_test, protocol_test) {
    auto obj2 = obj.protocol("tcp");
    ASSERT_EQ("tcp", obj.protocol());
    ASSERT_EQ(obj2, obj);
}

TEST_F(Endpoint_test, address_test) {
    auto obj2 = obj.address("127.0.0.1");
    ASSERT_EQ("127.0.0.1", obj.address());
    ASSERT_EQ(obj2.to_string(), obj.to_string());
}