#include"helix/core/io.hpp"
#include<sys/mman.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<unistd.h>

namespace helix {

FileReader::FileReader(const std::string& path) : path_(path)
{
    open(path);
}

FileReader::~FileReader()
{
    close();
}

void FileReader::open(const std::string& path)
{
    path_=path;
    file_.open(path,std::ios::binary);
    HELIX_CHECK(file_.is_open(),"failed to open file: "+path);
}

void FileReader::close()
{
    if(file_.is_open())
    {
        file_.close();
    }
}

void FileReader::seek(size_t pos)
{
    file_.seekg(pos);
}

size_t FileReader::tell()
{
    return file_.tellg();
}

size_t FileReader::size()
{
    auto pos=file_.tellg();
    file_.seekg(0,std::ios::end);
    auto sz=file_.tellg();
    file_.seekg(pos);
    return sz;
}

FileWriter::FileWriter(const std::string& path) : path_(path)
{
    open(path);
}

FileWriter::~FileWriter()
{
    close();
}

void FileWriter::open(const std::string& path)
{
    path_=path;
    file_.open(path,std::ios::binary);
    HELIX_CHECK(file_.is_open(),"failed to open file: "+path);
}

void FileWriter::close()
{
    if(file_.is_open())
    {
        file_.close();
    }
}

void FileWriter::flush()
{
    file_.flush();
}

MappedFile::~MappedFile()
{
    close();
}

void MappedFile::open(const std::string& path,bool readOnly)
{
    readOnly_=readOnly;
    
    int flags=readOnly ? O_RDONLY : O_RDWR;
    fd_=::open(path.c_str(),flags);
    HELIX_CHECK(fd_!=-1,"failed to open file: "+path);
    
    struct stat sb;
    HELIX_CHECK(fstat(fd_,&sb)!=-1,"failed to stat file: "+path);
    size_=sb.st_size;
    
    int prot=readOnly ? PROT_READ : (PROT_READ|PROT_WRITE);
    data_=mmap(nullptr,size_,prot,MAP_SHARED,fd_,0);
    HELIX_CHECK(data_!=MAP_FAILED,"failed to mmap file: "+path);
}

void MappedFile::close()
{
    if(data_!=nullptr && data_!=MAP_FAILED)
    {
        munmap(data_,size_);
        data_=nullptr;
    }
    
    if(fd_!=-1)
    {
        ::close(fd_);
        fd_=-1;
    }
}

}

