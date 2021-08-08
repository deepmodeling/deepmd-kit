#pragma once

#include <stdexcept>
#include <string>

namespace deepmd{
    struct
    deepmd_exception: public std::runtime_error {
    public:
        deepmd_exception(): runtime_error("DeePMD-kit Error!") {};
        deepmd_exception(const std::string& msg): runtime_error(std::string("DeePMD-kit Error: ") + msg) {};
    };

    struct
    deepmd_exception_oom: public std::runtime_error{
    public:
        deepmd_exception_oom(): runtime_error("DeePMD-kit OOM!") {};
        deepmd_exception_oom(const std::string& msg): runtime_error(std::string("DeePMD-kit OOM: ") + msg) {};
    };
};