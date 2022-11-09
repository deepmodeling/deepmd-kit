#pragma once

#include <stdexcept>
#include <string>

typedef double ENERGYTYPE;

namespace deepmd
{
    /**
     * @brief General DeePMD-kit exception. Throw if anything doesn't work.
     **/
    struct
        deepmd_exception : public std::runtime_error
    {
    public:
        deepmd_exception() : runtime_error("DeePMD-kit Error!"){};
        deepmd_exception(const std::string &msg) : runtime_error(std::string("DeePMD-kit Error: ") + msg){};
    };

    /**
     * @brief Convert pbtxt to pb.
     * @param[in] fn_pb_txt Filename of the pb txt file.
     * @param[in] fn_pb Filename of the pb file.
     **/
    void
    convert_pbtxt_to_pb(std::string fn_pb_txt, std::string fn_pb){
        DP_ConvertPbtxtToPb(fn_pb_txt.c_str(), fn_pb.c_str());
    };
}
