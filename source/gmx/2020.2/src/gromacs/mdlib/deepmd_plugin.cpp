#include <iostream>
#include <fstream>
#include "deepmd/DeepPot.h"
#include "deepmd_plugin.h"
#include "json.hpp"

deepmd_plugin* deepmdPlugin;
bool           useDeepmd;    

void init_deepmd () {
    deepmdPlugin = new deepmd_plugin;
    char*  json_file = getenv("GMX_DEEPMD_INPUT_JSON");
    if (json_file == NULL)
    {
        std::cout << "To use Deepmd in GROMACS, please set GMX_DEEPMD_INPUT_JSON environment variable" << std::endl;
        useDeepmd = false;
    }
    else
    {
        std::ifstream fp (json_file);
        if (fp.is_open())
        {
            std::cout << "Init deepmd plugin from: " << json_file << std::endl;

            deepmd::DeepPot       nnp;
            std::vector<int >     dtype;
            std::vector<int >     dindex;
            int                   natom;
            bool                  pbc;
            float                 lmd;

            nlohmann::json jdata;
            fp >> jdata;
            std::string graph_file = jdata["graph_file"];
            std::string type_file  = jdata["type_file"];
            std::string index_file = jdata["index_file"];

            /* lambda */
            if (jdata.contains("lambda"))
            {
                lmd = jdata["lambda"];
            }
            else
            {
                lmd = 1.0;
            }
            std::cout << "Setting lambda: " << lmd << std::endl;
            /* lambda */

            /* pbc */
            if (jdata.contains("pbc"))
            {
                pbc = jdata["pbc"];
            }
            else
            {
                pbc = true;
            }
            std::cout << "Setting pbc: " << pbc << std::endl;
            /* pbc */

            std::string              line;
            std::istringstream       iss;
            int                      val;

            /* read type file */
            std::ifstream            ft(type_file);
            if (ft.is_open())
            {
                getline(ft, line);
                iss.clear();
                iss.str(line);
                while (iss >> val)
                {
                    dtype.push_back(val);
                }
                natom = dtype.size();
                std::cout << "Number of atoms: " << natom << std::endl;
            }
            else
            {
                std::cerr << "Not found type file: " << type_file << std::endl;
                exit(1); 
            }
            /* read type file */

            /* read index file */
            std::ifstream  fi(index_file);
            if (fi.is_open())
            {
                getline(fi, line);
                iss.clear();
                iss.str(line);
                while (iss >> val)
                {
                    dindex.push_back(val);
                }
                if (dindex.size() != natom)
                {
                    std::cerr << "Number of atoms in index file (" << dindex.size() << ") does not match type file (" << natom << ")!" << std::endl;
                    exit(1);
                }
            }
            else
            {
                std::cerr << "Not found index file: " << index_file << std::endl;
                exit(1);
            }
            /* read index file */

            /* init model */
            std::cout << "Begin Init Model: " << graph_file << std::endl;
            nnp.init(graph_file);
            std::cout << "Successfully load model!" << std::endl;
            std::string summary;
            nnp.print_summary(summary);
            std::cout << "Summary: " << std::endl << summary << std::endl;
            std::string map;
            nnp.get_type_map(map);
            std::cout << "Atom map: " << map << std::endl;
            /* init model */

            deepmdPlugin->pbc       = pbc;
            deepmdPlugin->lmd       = lmd;
            deepmdPlugin->nnp       = nnp;
            deepmdPlugin->dtype     = dtype;
            deepmdPlugin->dindex    = dindex;
            deepmdPlugin->natom     = natom;
            useDeepmd               = true;
            std::cout << "Successfully init plugin!" << std::endl;
        }
        else
        {
            std::cerr << "Invaild json file: " << json_file << std::endl;
            exit(1);
        }
    }
}