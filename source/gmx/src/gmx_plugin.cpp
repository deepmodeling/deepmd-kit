// SPDX-License-Identifier: LGPL-3.0-or-later
#include "gmx_plugin.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "json.hpp"

using deepmd::DeepmdPlugin;

DeepmdPlugin::DeepmdPlugin() { nnp = new deepmd_compat::DeepPot; }

DeepmdPlugin::DeepmdPlugin(char* json_file) {
  nnp = new deepmd_compat::DeepPot;
  DeepmdPlugin::init_from_json(json_file);
}

DeepmdPlugin::~DeepmdPlugin() { delete nnp; }

void DeepmdPlugin::init_from_json(char* json_file) {
  std::ifstream fp(json_file);
  if (fp.is_open()) {
    std::cout << "Init deepmd plugin from: " << json_file << std::endl;
    nlohmann::json jdata;
    fp >> jdata;
    std::string graph_file = jdata["graph_file"];
    std::string type_file = jdata["type_file"];
    std::string index_file = jdata["index_file"];

    /* lambda */
    if (jdata.contains("lambda")) {
      DeepmdPlugin::lmd = jdata["lambda"];
    } else {
      DeepmdPlugin::lmd = 1.0;
    }
    std::cout << "Setting lambda: " << DeepmdPlugin::lmd << std::endl;
    /* lambda */

    /* pbc */
    if (jdata.contains("pbc")) {
      DeepmdPlugin::pbc = jdata["pbc"];
    } else {
      DeepmdPlugin::pbc = true;
    }
    std::cout << "Setting pbc: " << DeepmdPlugin::pbc << std::endl;
    /* pbc */

    std::string line;
    std::istringstream iss;
    int val;

    /* read type file */
    std::ifstream ft(type_file);
    if (ft.is_open()) {
      getline(ft, line);
      iss.clear();
      iss.str(line);
      while (iss >> val) {
        DeepmdPlugin::dtype.push_back(val);
      }
      DeepmdPlugin::natom = DeepmdPlugin::dtype.size();
      std::cout << "Number of atoms: " << DeepmdPlugin::natom << std::endl;
    } else {
      std::cerr << "Not found type file: " << type_file << std::endl;
      exit(1);
    }
    /* read type file */

    /* read index file */
    std::ifstream fi(index_file);
    if (fi.is_open()) {
      getline(fi, line);
      iss.clear();
      iss.str(line);
      while (iss >> val) {
        DeepmdPlugin::dindex.push_back(val);
      }
      if (DeepmdPlugin::dindex.size() != DeepmdPlugin::natom) {
        std::cerr << "Number of atoms in index file ("
                  << DeepmdPlugin::dindex.size()
                  << ") does not match type file (" << DeepmdPlugin::natom
                  << ")!" << std::endl;
        exit(1);
      }
    } else {
      std::cerr << "Not found index file: " << index_file << std::endl;
      exit(1);
    }
    /* read index file */

    /* init model */
    std::cout << "Begin Init Model: " << graph_file << std::endl;
    DeepmdPlugin::nnp->init(graph_file);
    std::cout << "Successfully load model!" << std::endl;
    std::string summary;
    DeepmdPlugin::nnp->print_summary(summary);
    std::cout << "Summary: " << std::endl << summary << std::endl;
    std::string map;
    DeepmdPlugin::nnp->get_type_map(map);
    std::cout << "Atom map: " << map << std::endl;
    /* init model */

    std::cout << "Successfully init plugin!" << std::endl;
  } else {
    std::cerr << "Invaild json file: " << json_file << std::endl;
    exit(1);
  }
}
