#pragma once

#include"helix/common/types.hpp"
#include"helix/common/utils.hpp"
#include<string>
#include<fstream>
#include<vector>

namespace helix {

class FileReader {
public:
    explicit FileReader(const std::string& path);
    ~FileReader();
    
    void open(const std::string& path);
    void close();
    bool isOpen() const { return file_.is_open(); }
    
    template<typename T>
    void read(T* data,size_t count)
    {
        HELIX_CHECK(file_.is_open(),"file not open");
        file_.read(reinterpret_cast<char*>(data),count*sizeof(T));
        HELIX_CHECK(file_.good(),"read failed");
    }
    
    void seek(size_t pos);
    size_t tell();
    size_t size();
    
private:
    std::ifstream file_;
    std::string path_;
};

class FileWriter {
public:
    explicit FileWriter(const std::string& path);
    ~FileWriter();
    
    void open(const std::string& path);
    void close();
    bool isOpen() const { return file_.is_open(); }
    
    template<typename T>
    void write(const T* data,size_t count)
    {
        HELIX_CHECK(file_.is_open(),"file not open");
        file_.write(reinterpret_cast<const char*>(data),count*sizeof(T));
        HELIX_CHECK(file_.good(),"write failed");
    }
    
    void flush();
    
private:
    std::ofstream file_;
    std::string path_;
};

class MappedFile {
public:
    MappedFile()=default;
    ~MappedFile();
    
    void open(const std::string& path,bool readOnly=true);
    void close();
    
    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }
    
private:
    void* data_=nullptr;
    size_t size_=0;
    int fd_=-1;
    bool readOnly_=true;
};

}

