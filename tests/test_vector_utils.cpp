#include<gtest/gtest.h>
#include"helix/core/vector_utils.hpp"
#include<cmath>

TEST(VectorUtilsTest,VectorNorm) {
    float vec[]={3.0f,4.0f};
    float norm=helix::vectorNorm(vec,2);
    EXPECT_NEAR(norm,5.0f,1e-5f);
}

TEST(VectorUtilsTest,NormalizeVector) {
    float vec[]={3.0f,4.0f};
    helix::normalizeVector(vec,2);
    
    EXPECT_NEAR(vec[0],0.6f,1e-5f);
    EXPECT_NEAR(vec[1],0.8f,1e-5f);
    
    float norm=helix::vectorNorm(vec,2);
    EXPECT_NEAR(norm,1.0f,1e-5f);
}

TEST(VectorUtilsTest,NormalizeVectors) {
    float vecs[]={3.0f,4.0f,5.0f,12.0f};
    helix::normalizeVectors(vecs,2,2);
    
    float norm1=helix::vectorNorm(vecs,2);
    float norm2=helix::vectorNorm(vecs+2,2);
    
    EXPECT_NEAR(norm1,1.0f,1e-5f);
    EXPECT_NEAR(norm2,1.0f,1e-5f);
}

TEST(VectorUtilsTest,Vector2D) {
    helix::Vector2D<float> vec(10,5);
    
    EXPECT_EQ(vec.rows(),10);
    EXPECT_EQ(vec.cols(),5);
    EXPECT_EQ(vec.size(),50);
    
    vec[0][0]=1.0f;
    EXPECT_EQ(vec[0][0],1.0f);
    
    vec.resize(20,10);
    EXPECT_EQ(vec.rows(),20);
    EXPECT_EQ(vec.cols(),10);
}

TEST(VectorUtilsTest,CopyVectors) {
    float src[]={1.0f,2.0f,3.0f,4.0f};
    float dst[4];
    
    helix::copyVectors(dst,src,2,2);
    
    for(int i=0;i<4;++i)
    {
        EXPECT_EQ(dst[i],src[i]);
    }
}

