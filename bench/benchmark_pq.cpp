#include"helix/index/index_pq.hpp"
#include"helix/index/index_flat.hpp"
#include"helix/benchmark/dataset.hpp"
#include"helix/benchmark/metrics.hpp"
#include<iostream>
#include<chrono>

int main(int,char**)
{
    std::cout<<"Helix IndexPQ Benchmark"<<std::endl;
    std::cout<<"======================="<<std::endl<<std::endl;
    
    const int dim=128;
    const int ntrain=10000;
    const int nbase=10000;
    const int nquery=100;
    const int k=100;
    const int nsub=8;
    const int nbits=8;
    
    std::cout<<"Generating synthetic dataset..."<<std::endl;
    helix::BenchmarkDataset dataset;
    dataset.generateSynthetic(ntrain,nbase,nquery,dim);
    std::cout<<"  Train: "<<dataset.train.size()<<" vectors"<<std::endl;
    std::cout<<"  Base: "<<dataset.database.size()<<" vectors"<<std::endl;
    std::cout<<"  Query: "<<dataset.queries.size()<<" vectors"<<std::endl;
    std::cout<<std::endl;
    
    helix::IndexConfig config(dim,helix::MetricType::L2,helix::IndexType::PQ);
    helix::IndexPQ index(config,nsub,nbits);
    
    std::cout<<"Training quantizer..."<<std::endl;
    auto t0=std::chrono::high_resolution_clock::now();
    index.train(dataset.train.data(),dataset.train.size());
    auto t1=std::chrono::high_resolution_clock::now();
    double trainTime=std::chrono::duration<double,std::milli>(t1-t0).count();
    std::cout<<"  Train time: "<<trainTime<<" ms"<<std::endl;
    std::cout<<std::endl;
    
    std::cout<<"Adding vectors..."<<std::endl;
    t0=std::chrono::high_resolution_clock::now();
    index.add(dataset.database.data(),dataset.database.size());
    t1=std::chrono::high_resolution_clock::now();
    double addTime=std::chrono::duration<double,std::milli>(t1-t0).count();
    std::cout<<"  Add time: "<<addTime<<" ms"<<std::endl;
    std::cout<<"  Vectors in index: "<<index.ntotal()<<std::endl;
    std::cout<<std::endl;
    
    std::cout<<"Running search benchmark..."<<std::endl;
    helix::MetricsCollector collector;
    
    std::vector<helix::SearchResults> results(nquery);
    collector.startTimer();
    
    for(int i=0;i<nquery;++i)
    {
        collector.recordQuery();
        results[i]=index.search(dataset.queries.data()+i*dim,k);
        collector.stopTimer();
    }
    
    helix::BenchmarkResults benchResults;
    benchResults.recall=collector.computeRecall(results,dataset.groundtruth.labels(),nquery,100);
    benchResults.latency=collector.computeLatency();
    benchResults.throughput=collector.computeThroughput();
    benchResults.memory_bytes=index.ntotal()*nsub;
    
    std::cout<<std::endl;
    benchResults.print();
    
    std::string jsonPath="/tmp/helix_benchmark_pq.json";
    benchResults.saveJson(jsonPath);
    std::cout<<"\nResults saved to: "<<jsonPath<<std::endl;
    
    return 0;
}

