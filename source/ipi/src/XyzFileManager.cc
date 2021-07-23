#include "StringSplit.h"
#include "XyzFileManager.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <assert.h>

void
XyzFileManager::
read (const std::string & file,
      std::vector<std::string > & atom_name,
      std::vector<std::vector<double > > & posi,
      std::vector<std::vector<double > > & velo,
      std::vector<std::vector<double > > & forc)
{
  // getBoxSize (file, boxsize);
  
  posi.clear();
  velo.clear();

  std::ifstream data0 (file.c_str());
  if (!data0.is_open()) {
    std::cerr <<  "cannot open file " << file << std::endl;
    exit(1);
  }
  
  std::string valueline;
  std::vector<std::string> words;
  words.reserve (10);
  std::string tmpname;
  std::vector<double > tmpp(3);
  std::vector<double > tmpv(3);
  std::vector<double > tmpf(3);
  std::getline(data0, valueline);
  long long int numb_atom = atoll (valueline.c_str());
  std::getline(data0, valueline);
  
  for (long long int ii = 0; ii< numb_atom; ++ii) {
    std::getline(data0, valueline);
    StringOperation::split (std::string(valueline), words);
    if (words.size() == 10){
      tmpp[0] = atof (words[1+0].c_str());
      tmpp[1] = atof (words[1+1].c_str());
      tmpp[2] = atof (words[1+2].c_str());
      tmpv[0] = atof (words[1+3].c_str());
      tmpv[1] = atof (words[1+4].c_str());
      tmpv[2] = atof (words[1+5].c_str());
      tmpf[0] = atof (words[1+6].c_str());
      tmpf[1] = atof (words[1+7].c_str());
      tmpf[2] = atof (words[1+8].c_str());
      posi.push_back (tmpp);
      velo.push_back (tmpv);
      forc.push_back (tmpf);
      atom_name.push_back (words[0]);
    }
    else if (words.size() == 7){
      tmpp[0] = atof (words[1+0].c_str());
      tmpp[1] = atof (words[1+1].c_str());
      tmpp[2] = atof (words[1+2].c_str());
      tmpv[0] = atof (words[1+3].c_str());
      tmpv[1] = atof (words[1+4].c_str());
      tmpv[2] = atof (words[1+5].c_str());
      posi.push_back (tmpp);
      velo.push_back (tmpv);
      atom_name.push_back (words[0]);
    }
    else if (words.size() == 4){
      tmpp[0] = atof (words[1+0].c_str());
      tmpp[1] = atof (words[1+1].c_str());
      tmpp[2] = atof (words[1+2].c_str());
      posi.push_back (tmpp);
      atom_name.push_back (words[0]);
    }
    else {
      std::cerr << "XyzFileManager::read: wrong format, line has "<< words.size() << " words" << std::endl;
      exit (1);
    }
  }
}


