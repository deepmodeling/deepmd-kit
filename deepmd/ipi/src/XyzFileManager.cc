#include "StringSplit.h"
#include "XyzFileManager.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <assert.h>

void
XyzFileManager::
read (const string & file,
      vector<string > & atom_name,
      vector<vector<double > > & posi,
      vector<vector<double > > & velo,
      vector<vector<double > > & forc)
{
  // getBoxSize (file, boxsize);
  
  posi.clear();
  velo.clear();

  ifstream data0 (file.c_str());
  if (!data0.is_open()) {
    cerr <<  "cannot open file " << file << endl;
    exit(1);
  }
  
  string valueline;
  vector<string> words;
  words.reserve (10);
  string tmpname;
  vector<double > tmpp(3);
  vector<double > tmpv(3);
  vector<double > tmpf(3);
  std::getline(data0, valueline);
  long long int numb_atom = atoll (valueline.c_str());
  std::getline(data0, valueline);
  
  for (long long int ii = 0; ii< numb_atom; ++ii) {
    std::getline(data0, valueline);
    StringOperation::split (string(valueline), words);
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
      cerr << "XyzFileManager::read: wrong format, line has "<< words.size() << " words" << endl;
      exit (1);
    }
  }
}


