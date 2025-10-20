#include<gtest/gtest.h>
#include"helix/core/manifest.hpp"
#include"helix/core/io.hpp"

TEST(ManifestTest,Roundtrip)
{
    helix::Manifest m;
    m.version="0.1.0";
    m.indexType="IVF";
    m.dimension=128;
    m.metric="L2";

    std::string path="/tmp/manifest_test.bin";
    helix::ManifestIO::write(m,path);

    helix::Manifest r=helix::ManifestIO::read(path);
    EXPECT_EQ(r.version,m.version);
    EXPECT_EQ(r.indexType,m.indexType);
    EXPECT_EQ(r.dimension,m.dimension);
    EXPECT_EQ(r.metric,m.metric);
}

TEST(ManifestTest,EmptyStrings)
{
    helix::Manifest m;
    m.dimension=0;

    std::string path="/tmp/manifest_empty.bin";
    helix::ManifestIO::write(m,path);
    helix::Manifest r=helix::ManifestIO::read(path);

    EXPECT_TRUE(r.version.empty());
    EXPECT_TRUE(r.indexType.empty());
    EXPECT_TRUE(r.metric.empty());
    EXPECT_EQ(r.dimension,0);
}
