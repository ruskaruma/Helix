#include<gtest/gtest.h>
#include"helix/quantization/product_quantizer.hpp"
#include"helix/common/utils.hpp"
#include<vector>
#include<random>
#include<cmath>

TEST(ProductQuantizerTest,Construction) {
    helix::ProductQuantizer pq(128,8,8);
    
    EXPECT_EQ(pq.dimension(),128);
    EXPECT_EQ(pq.nsub(),8);
    EXPECT_EQ(pq.nbits(),8);
    EXPECT_EQ(pq.ksub(),256);
    EXPECT_EQ(pq.dsub(),16);
    EXPECT_FALSE(pq.isTrained());
}

TEST(ProductQuantizerTest,DimensionValidation) {
    EXPECT_THROW(helix::ProductQuantizer(127,8,8),helix::HelixException);
}

TEST(ProductQuantizerTest,TrainAndEncode) {
    helix::ProductQuantizer pq(64,4,6);
    
    std::vector<float> vectors(1000*64);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f,1.0f);
    
    for(auto& v : vectors)
    {
        v=dist(rng);
    }
    
    pq.train(vectors.data(),1000);
    EXPECT_TRUE(pq.isTrained());
    
    std::vector<uint8_t> codes(1000*pq.nsub());
    pq.encode(vectors.data(),1000,codes.data());
    
    for(size_t i=0;i<codes.size();++i)
    {
        EXPECT_LT(codes[i],pq.ksub());
    }
}

TEST(ProductQuantizerTest,EncodeDecodeRoundtrip) {
    helix::ProductQuantizer pq(32,4,4);
    
    std::vector<float> vectors(100*32);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f,1.0f);
    
    for(auto& v : vectors)
    {
        v=dist(rng);
    }
    
    pq.train(vectors.data(),100);
    
    std::vector<uint8_t> codes(100*pq.nsub());
    pq.encode(vectors.data(),100,codes.data());
    
    std::vector<float> decoded(100*32);
    pq.decode(codes.data(),100,decoded.data());
    
    for(int i=0;i<100;++i)
    {
        float error=0.0f;
        for(int d=0;d<32;++d)
        {
            float diff=vectors[i*32+d]-decoded[i*32+d];
            error+=diff*diff;
        }
        error=std::sqrt(error);
        EXPECT_GT(error,0.0f);
    }
}

TEST(ProductQuantizerTest,AsymmetricDistance) {
    helix::ProductQuantizer pq(16,2,4);
    
    std::vector<float> vectors(100*16);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f,1.0f);
    
    for(auto& v : vectors)
    {
        v=dist(rng);
    }
    
    pq.train(vectors.data(),100);
    
    std::vector<uint8_t> codes(100*pq.nsub());
    pq.encode(vectors.data(),100,codes.data());
    
    std::vector<float> query(16);
    for(auto& q : query)
    {
        q=dist(rng);
    }
    
    float dist1=pq.computeAsymmetricDistance(query.data(),codes.data());
    EXPECT_GE(dist1,0.0f);
}

TEST(ProductQuantizerTest,DistanceTable) {
    helix::ProductQuantizer pq(32,4,4);
    
    std::vector<float> vectors(100*32);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f,1.0f);
    
    for(auto& v : vectors)
    {
        v=dist(rng);
    }
    
    pq.train(vectors.data(),100);
    
    std::vector<float> query(32);
    for(auto& q : query)
    {
        q=dist(rng);
    }
    
    std::vector<float> distTable(pq.nsub()*pq.ksub());
    pq.computeDistanceTable(query.data(),distTable.data());
    
    EXPECT_EQ(distTable.size(),pq.nsub()*pq.ksub());
    
    for(float d : distTable)
    {
        EXPECT_GE(d,0.0f);
    }
}

TEST(ProductQuantizerTest,SmallQuantizer) {
    helix::ProductQuantizer pq(8,2,2);
    
    std::vector<float> vectors(20*8);
    for(size_t i=0;i<vectors.size();++i)
    {
        vectors[i]=static_cast<float>(i%10);
    }
    
    pq.train(vectors.data(),20);
    EXPECT_TRUE(pq.isTrained());
    
    std::vector<uint8_t> codes(20*2);
    pq.encode(vectors.data(),20,codes.data());
    
    for(auto c : codes)
    {
        EXPECT_LT(c,4);
    }
}

