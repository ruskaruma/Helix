#pragma once

#include<string>
#include<stdexcept>
#include<sstream>

namespace helix {

class HelixException : public std::runtime_error {
public:
    explicit HelixException(const std::string& msg) : std::runtime_error(msg) {}
};

#define HELIX_CHECK(condition,message) \
    do { \
        if(!(condition)) { \
            std::ostringstream oss; \
            oss<<__FILE__<<":"<<__LINE__<<" "<<message; \
            throw HelixException(oss.str()); \
        } \
    } while(0)

#define HELIX_CHECK_EQ(a,b,message) HELIX_CHECK((a)==(b),message)
#define HELIX_CHECK_NE(a,b,message) HELIX_CHECK((a)!=(b),message)
#define HELIX_CHECK_GE(a,b,message) HELIX_CHECK((a)>=(b),message)
#define HELIX_CHECK_GT(a,b,message) HELIX_CHECK((a)>(b),message)
#define HELIX_CHECK_LE(a,b,message) HELIX_CHECK((a)<=(b),message)
#define HELIX_CHECK_LT(a,b,message) HELIX_CHECK((a)<(b),message)

}

