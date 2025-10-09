#include<gtest/gtest.h>
#include"helix/helix.hpp"

TEST(VersionTest,GetVersion) {
    const char* version=helix::getVersion();
    ASSERT_NE(version,nullptr);
    EXPECT_STREQ(version,"0.1.0");
}

