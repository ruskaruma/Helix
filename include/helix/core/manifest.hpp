#pragma once

#include"helix/common/types.hpp"
#include"helix/common/utils.hpp"
#include<string>
#include<vector>

namespace helix {

struct Manifest
{
    std::string version;
    std::string indexType;
    dim_t dimension;
    std::string metric;

    Manifest() : dimension(0) {}
};

class ManifestIO
{
  public:
    static void write(const Manifest& m,const std::string& path);
    static Manifest read(const std::string& path);
};

} // namespace helix
