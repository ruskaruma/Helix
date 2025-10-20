#include"helix/core/manifest.hpp"
#include"helix/core/io.hpp"

namespace helix {

static void writeString(FileWriter& w,const std::string& s)
{
    idx_t n=(idx_t)s.size();
    w.write(&n,1);
    if(n>0) { w.write(reinterpret_cast<const char*>(s.data()),(size_t)n); }
}

static std::string readString(FileReader& r)
{
    idx_t n=0;
    r.read(&n,1);
    std::string s;
    if(n>0)
    {
        s.resize((size_t)n);
        r.read(reinterpret_cast<char*>(s.data()),(size_t)n);
    }
    return s;
}

void ManifestIO::write(const Manifest& m,const std::string& path)
{
    FileWriter w(path);
    writeString(w,m.version);
    writeString(w,m.indexType);
    w.write(&m.dimension,1);
    writeString(w,m.metric);
    w.close();
}

Manifest ManifestIO::read(const std::string& path)
{
    FileReader r(path);
    Manifest m;
    m.version=readString(r);
    m.indexType=readString(r);
    r.read(&m.dimension,1);
    m.metric=readString(r);
    r.close();
    return m;
}

} 