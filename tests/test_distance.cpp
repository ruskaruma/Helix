#include<gtest/gtest.h>
#include"helix/core/distance.hpp"
#include<vector>
#include<cmath>

TEST(DistanceTest,L2DistanceBasic) {
    helix::L2Distance dist;
    
    float a[]={1.0f,2.0f,3.0f};
    float b[]={4.0f,5.0f,6.0f};
    
    float result=dist.compute(a,b,3);
    float expected=9.0f+9.0f+9.0f;
    EXPECT_NEAR(result,expected,1e-5f);
}

TEST(DistanceTest,L2DistanceSame) {
    helix::L2Distance dist;
    
    float a[]={1.0f,2.0f,3.0f};
    
    float result=dist.compute(a,a,3);
    EXPECT_NEAR(result,0.0f,1e-5f);
}

TEST(DistanceTest,InnerProductDistance) {
    helix::InnerProductDistance dist;
    
    float a[]={1.0f,2.0f,3.0f};
    float b[]={4.0f,5.0f,6.0f};
    
    float result=dist.compute(a,b,3);
    float expected=-(4.0f+10.0f+18.0f);
    EXPECT_NEAR(result,expected,1e-5f);
}

TEST(DistanceTest,CosineDistance) {
    helix::CosineDistance dist;
    
    float a[]={1.0f,0.0f,0.0f};
    float b[]={0.0f,1.0f,0.0f};
    
    float result=dist.compute(a,b,3);
    EXPECT_NEAR(result,1.0f,1e-5f);
    
    float c[]={1.0f,0.0f,0.0f};
    float d[]={1.0f,0.0f,0.0f};
    result=dist.compute(c,d,3);
    EXPECT_NEAR(result,0.0f,1e-5f);
}

TEST(DistanceTest,CreateDistanceComputer) {
    auto l2=helix::createDistanceComputer(helix::MetricType::L2);
    ASSERT_NE(l2,nullptr);
    
    auto ip=helix::createDistanceComputer(helix::MetricType::InnerProduct);
    ASSERT_NE(ip,nullptr);
    
    auto cosine=helix::createDistanceComputer(helix::MetricType::Cosine);
    ASSERT_NE(cosine,nullptr);
}

TEST(DistanceTest,BatchCompute) {
    helix::L2Distance dist;
    
    float queries[]={1.0f,2.0f,3.0f,4.0f};
    float database[]={1.0f,2.0f,5.0f,6.0f};
    float distances[4];
    
    dist.computeBatch(queries,database,2,2,2,distances);
    
    EXPECT_NEAR(distances[0],0.0f,1e-5f);
    EXPECT_NEAR(distances[1],16.0f+16.0f,1e-5f);
}

