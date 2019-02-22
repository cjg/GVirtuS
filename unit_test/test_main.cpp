#include <gtest/gtest.h>
#include <gmock/gmock.h>

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    auto value = RUN_ALL_TESTS();

    return 0;
}
