#ifndef _GMX_PLUGIN_H_
#define _GMX_PLUGIN_H_
#include "DeepPot.h"

namespace deepmd
{

class DeepmdPlugin
{
    public:
        DeepmdPlugin();
        DeepmdPlugin(char*);
        ~DeepmdPlugin();  
        void              init_from_json(char*);
        deepmd::DeepPot*  nnp;
        std::vector<int > dtype;
        std::vector<int > dindex;
        bool              pbc;
        float             lmd;
        int               natom;
};

}

const float c_dp2gmx = 0.1;
const float e_dp2gmx = 96.48533132;
const float f_dp2gmx = 964.8533132;
#endif