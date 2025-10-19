#include<gtest/gtest.h>
#include"helix/core/kmeans.hpp"
#include"helix/common/utils.hpp"
#include<random>

class KMeansTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        dim=128;
        n=1000;
        k=8;
        data.resize(n*dim);
        
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f,1.0f);
        for(int i=0;i<n*dim;++i)
        {
            data[i]=dist(rng);
        }
    }

    int dim;
    int n;
    int k;
    std::vector<float> data;
};

TEST_F(KMeansTest,BasicTraining)
{
    helix::KMeans kmeans(k);
    kmeans.train(data.data(),n,dim);
    
    EXPECT_TRUE(kmeans.isTrained());
    EXPECT_EQ(kmeans.getCentroids().size(),k*dim);
}

TEST_F(KMeansTest,Assignment)
{
    helix::KMeans kmeans(k);
    kmeans.train(data.data(),n,dim);
    
    std::vector<helix::idx_t> assignments(n);
    for(int i=0;i<n;++i)
    {
        assignments[i]=kmeans.assign(data.data()+i*dim,dim);
        EXPECT_GE(assignments[i],0);
        EXPECT_LT(assignments[i],k);
    }
}

TEST_F(KMeansTest,Reset)
{
    helix::KMeans kmeans(k);
    kmeans.train(data.data(),n,dim);
    EXPECT_TRUE(kmeans.isTrained());
    
    kmeans.reset();
    EXPECT_FALSE(kmeans.isTrained());
    EXPECT_TRUE(kmeans.getCentroids().empty());
}

TEST_F(KMeansTest,InvalidInputs)
{
    helix::KMeans kmeans(k);
    
    EXPECT_THROW(kmeans.train(nullptr,n,dim),helix::HelixException);
    EXPECT_THROW(kmeans.train(data.data(),0,dim),helix::HelixException);
    EXPECT_THROW(kmeans.train(data.data(),n,0),helix::HelixException);
    EXPECT_THROW(kmeans.train(data.data(),k-1,dim),helix::HelixException);
}

TEST_F(KMeansTest,UntrainedAssignment)
{
    helix::KMeans kmeans(k);
    EXPECT_THROW(kmeans.assign(data.data(),dim),helix::HelixException);
}

TEST_F(KMeansTest,DimensionMismatch)
{
    helix::KMeans kmeans(k);
    kmeans.train(data.data(),n,dim);
    EXPECT_THROW(kmeans.assign(data.data(),dim+1),helix::HelixException);
}
