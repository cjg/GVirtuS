#include <util/JSON.h>
#include <backend/Property.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using testing::Eq;

namespace {
    class Json_test : public testing::Test {
    public:
        gvirtus::util::JSON<gvirtus::Property> obj;
        std::filesystem::path uri;

        Json_test() {
            obj;
        }
    };
}

TEST_F(Json_test, path_test) {
    namespace fs = std::filesystem;

    auto p = fs::current_path().parent_path();
    fs::path etc{"etc/"};
    fs::path resource{"properties.json"};
    uri = p / etc / resource;

    obj.path(uri);
    ASSERT_EQ(uri, obj.path());
}

TEST_F(Json_test, parser_test) {
    namespace fs = std::filesystem;

    auto p = fs::current_path().parent_path();
    fs::path etc{"etc/"};
    fs::path resource{"properties.json"};
    uri = p / etc / resource;

    obj.path(uri);
    auto from_json_to_property = obj.parser();
    ASSERT_FALSE(from_json_to_property.endpoints().empty());
    ASSERT_FALSE(from_json_to_property.plugins().empty());
}