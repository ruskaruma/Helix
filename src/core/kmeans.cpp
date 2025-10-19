#include"helix/core/kmeans.hpp"
#include"helix/core/distance.hpp"
#include"helix/common/utils.hpp"
#include<algorithm>
#include<numeric>
#include<cstring>

namespace helix {

KMeans::KMeans(int k,int maxIter)
    : k_(k),maxIter_(maxIter),dim_(0),trained_(false),rng_(std::random_device{}())
{
    HELIX_CHECK(k_>0,"k must be positive");
    HELIX_CHECK(maxIter_>0,"maxIter must be positive");
}

void KMeans::train(const float* data,idx_t n,dim_t dim)
{
    HELIX_CHECK(data!=nullptr,"data cannot be null");
    HELIX_CHECK(n>0,"n must be positive");
    HELIX_CHECK(dim>0,"dim must be positive");
    HELIX_CHECK(n>=k_,"n must be >= k");

    dim_=dim;
    centroids_.resize(k_*dim_);
    trained_=false;

    initializeCentroids(data,n);

    std::vector<idx_t> assignments(n);
    std::vector<int> clusterSizes(k_);

    for(int iter=0;iter<maxIter_;++iter)
    {
        bool changed=false;

        //assign points to clusters
        for(idx_t i=0;i<n;++i)
        {
            idx_t bestCluster=0;
            float bestDist=computeDistance(data+i*dim_,centroids_.data());

            for(int j=1;j<k_;++j)
            {
                float dist=computeDistance(data+i*dim_,centroids_.data()+j*dim_);
                if(dist<bestDist)
                {
                    bestDist=dist;
                    bestCluster=j;
                }
            }

            if(assignments[i]!=bestCluster)
            {
                assignments[i]=bestCluster;
                changed=true;
            }
        }

        if(!changed)
        {
            break;
        }

        updateCentroids(data,n,assignments);
    }

    trained_=true;
}

idx_t KMeans::assign(const float* vector,dim_t dim) const
{
    HELIX_CHECK(trained_,"kmeans not trained");
    HELIX_CHECK(vector!=nullptr,"vector cannot be null");
    HELIX_CHECK(dim==dim_,"dimension mismatch");

    idx_t bestCluster=0;
    float bestDist=computeDistance(vector,centroids_.data());

    for(int i=1;i<k_;++i)
    {
        float dist=computeDistance(vector,centroids_.data()+i*dim_);
        if(dist<bestDist)
        {
            bestDist=dist;
            bestCluster=i;
        }
    }

    return bestCluster;
}

void KMeans::setCentroids(const std::vector<float>& centroids)
{
    centroids_=centroids;
    trained_=true;
    dim_=centroids.size()/k_;
}

void KMeans::reset()
{
    centroids_.clear();
    trained_=false;
    dim_=0;
}

void KMeans::initializeCentroids(const float* data,idx_t n)
{
    //kmeans++ initialization
    std::vector<bool> used(n,false);
    std::uniform_int_distribution<idx_t> dist(0,n-1);

    //first centroid: random point
    idx_t first=dist(rng_);
    std::memcpy(centroids_.data(),data+first*dim_,dim_*sizeof(float));
    used[first]=true;

    for(int i=1;i<k_;++i)
    {
        std::vector<float> distances(n);
        float totalDist=0.0f;

        for(idx_t j=0;j<n;++j)
        {
            if(used[j])
            {
                distances[j]=0.0f;
                continue;
            }

            float minDist=std::numeric_limits<float>::max();
            for(int k=0;k<i;++k)
            {
                float d=computeDistance(data+j*dim_,centroids_.data()+k*dim_);
                minDist=std::min(minDist,d);
            }
            distances[j]=minDist*minDist;
            totalDist+=distances[j];
        }

        if(totalDist==0.0f)
        {
            break;
        }

        std::uniform_real_distribution<float> probDist(0.0f,totalDist);
        float target=probDist(rng_);
        float cumsum=0.0f;

        for(idx_t j=0;j<n;++j)
        {
            cumsum+=distances[j];
            if(cumsum>=target)
            {
                std::memcpy(centroids_.data()+i*dim_,data+j*dim_,dim_*sizeof(float));
                used[j]=true;
                break;
            }
        }
    }
}

void KMeans::updateCentroids(const float* data,idx_t n,const std::vector<idx_t>& assignments)
{
    std::vector<int> clusterSizes(k_,0);
    std::fill(centroids_.begin(),centroids_.end(),0.0f);

    for(idx_t i=0;i<n;++i)
    {
        int cluster=assignments[i];
        clusterSizes[cluster]++;
        for(dim_t d=0;d<dim_;++d)
        {
            centroids_[cluster*dim_+d]+=data[i*dim_+d];
        }
    }

    for(int i=0;i<k_;++i)
    {
        if(clusterSizes[i]>0)
        {
            for(dim_t d=0;d<dim_;++d)
            {
                centroids_[i*dim_+d]/=clusterSizes[i];
            }
        }
    }
}

float KMeans::computeDistance(const float* a,const float* b) const
{
    float sum=0.0f;
    for(dim_t i=0;i<dim_;++i)
    {
        float diff=a[i]-b[i];
        sum+=diff*diff;
    }
    return sum;
}

} // namespace helix
