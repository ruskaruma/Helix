#include<gtest/gtest.h>
#include"helix/benchmark/dataset.hpp"
#include<fstream>
#include<cmath>

TEST(DatasetTest,GenerateRandom) {
    helix::Dataset dataset;
    dataset.generateRandom(100,64,0.0f,1.0f,42);
    
    EXPECT_EQ(dataset.size(),100);
    EXPECT_EQ(dataset.dimension(),64);
    EXPECT_NE(dataset.data(),nullptr);
    
    for(int i=0;i<100*64;++i)
    {
        EXPECT_GE(dataset.data()[i],0.0f);
        EXPECT_LE(dataset.data()[i],1.0f);
    }
}

TEST(DatasetTest,GenerateGaussian) {
    helix::Dataset dataset;
    dataset.generateGaussian(100,64,0.0f,1.0f,42);
    
    EXPECT_EQ(dataset.size(),100);
    EXPECT_EQ(dataset.dimension(),64);
    
    float sum=0.0f;
    for(int i=0;i<100*64;++i)
    {
        sum+=dataset.data()[i];
    }
    float mean=sum/(100*64);
    EXPECT_NEAR(mean,0.0f,0.2f);
}

TEST(DatasetTest,Normalize) {
    helix::Dataset dataset;
    dataset.generateRandom(10,8,1.0f,2.0f,42);
    dataset.normalize();
    
    for(int i=0;i<10;++i)
    {
        float norm=0.0f;
        for(int j=0;j<8;++j)
        {
            float val=dataset.data()[i*8+j];
            norm+=val*val;
        }
        EXPECT_NEAR(std::sqrt(norm),1.0f,1e-5f);
    }
}

TEST(DatasetTest,FvecsWriteRead) {
    helix::Dataset dataset1;
    dataset1.generateRandom(50,32,0.0f,1.0f,42);
    
    std::string path="/tmp/helix_test.fvecs";
    std::ofstream file(path,std::ios::binary);
    
    for(int i=0;i<50;++i)
    {
        int dim=32;
        file.write(reinterpret_cast<const char*>(&dim),sizeof(int));
        file.write(reinterpret_cast<const char*>(dataset1.data()+i*32),32*sizeof(float));
    }
    file.close();
    
    helix::Dataset dataset2;
    dataset2.loadFvecs(path);
    
    EXPECT_EQ(dataset2.size(),50);
    EXPECT_EQ(dataset2.dimension(),32);
    
    for(int i=0;i<50*32;++i)
    {
        EXPECT_FLOAT_EQ(dataset2.data()[i],dataset1.data()[i]);
    }
}

TEST(DatasetTest,IvecsWriteRead) {
    std::string path="/tmp/helix_test.ivecs";
    std::ofstream file(path,std::ios::binary);
    
    std::vector<int> labels={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14};
    
    for(int i=0;i<3;++i)
    {
        int k=5;
        file.write(reinterpret_cast<const char*>(&k),sizeof(int));
        file.write(reinterpret_cast<const char*>(labels.data()+i*5),5*sizeof(int));
    }
    file.close();
    
    helix::Dataset dataset;
    dataset.loadIvecs(path);
    
    EXPECT_EQ(dataset.size(),3);
    EXPECT_EQ(dataset.dimension(),5);
    
    for(int i=0;i<15;++i)
    {
        EXPECT_EQ(dataset.labels()[i],labels[i]);
    }
}

TEST(DatasetTest,BenchmarkDatasetSynthetic) {
    helix::BenchmarkDataset dataset;
    dataset.generateSynthetic(100,500,10,64);
    
    EXPECT_EQ(dataset.train.size(),100);
    EXPECT_EQ(dataset.database.size(),500);
    EXPECT_EQ(dataset.queries.size(),10);
    EXPECT_EQ(dataset.groundtruth.size(),10);
    EXPECT_EQ(dataset.groundtruth.dimension(),100);
}

TEST(DatasetTest,EmptyDataset) {
    helix::Dataset dataset;
    EXPECT_EQ(dataset.size(),0);
    EXPECT_EQ(dataset.dimension(),0);
}

TEST(DatasetTest,RandomDifferentSeeds) {
    helix::Dataset dataset1,dataset2;
    dataset1.generateRandom(10,8,0.0f,1.0f,42);
    dataset2.generateRandom(10,8,0.0f,1.0f,43);
    
    bool different=false;
    for(int i=0;i<10*8;++i)
    {
        if(dataset1.data()[i]!=dataset2.data()[i])
        {
            different=true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

