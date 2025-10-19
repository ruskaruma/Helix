#include "helix/index/index_flat.hpp"
#include <iostream>
#include <random>
#include <vector>

int main()
{
    const int dim= 128;
    const int n= 1000;
    const int nq= 10;
    const int k= 5;

    helix::IndexConfig config(dim, helix::MetricType::L2, helix::IndexType::Flat);
    helix::IndexFlat index(config);

    std::vector<float> database(n * dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for(auto &v : database)
    {
        v= dist(rng);
    }

    index.train(database.data(), n);
    index.add(database.data(), n);

    std::cout << "Added " << index.ntotal() << " vectors" << std::endl;

    std::vector<float> queries(nq * dim);
    for(auto &q : queries)
    {
        q= dist(rng);
    }

    std::vector<helix::SearchResults> results;
    index.searchBatch(queries.data(), nq, k, results);

    std::cout << "Search completed for " << results.size() << " queries" << std::endl;
    std::cout << "Top-" << k << " results for query 0:" << std::endl;

    for(int i= 0; i < k; ++i)
    {
        std::cout << "  ID: " << results[0].results[i].id
                  << ", Distance: " << results[0].results[i].distance << std::endl;
    }

    return 0;
}
