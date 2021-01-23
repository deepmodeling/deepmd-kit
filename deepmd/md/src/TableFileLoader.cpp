#include "TableFileLoader.h"
#include "StringSplit.h"

#include <iostream>
#include <algorithm>

#define MaxLineLength 65536

using namespace std;

TableFileLoader::
TableFileLoader	(const char * file)
    :
    every (1)
{
  reinit (file);
}

unsigned
TableFileLoader::
getNumbColumns ()
{
  char valueline [MaxLineLength];  
  while (data.getline(valueline, MaxLineLength)){
    if (valueline[0] == '#' || valueline[0] == '@'){
      continue;
    }
    break;
  }  
  if (data.eof()){
    return 0;
  }
  else if (! data.good()){
    cerr << "error file reading state!" << endl;
    throw;
  }
  vector<string > words;
  StringOperation::split (string(valueline), words);

  data.close();
  reinit (file.c_str());
  return words.size();
}

void 
TableFileLoader::
reinit (const char * file_)
{
  file = string(file_);
  data.open (file.c_str());
  if (!data){
    cerr << "cannot open file \"" << file << "\"" << endl;
    throw;
  }
  count_read = 0;
  // inter_cols.push_back (0);
}

void 
TableFileLoader::
setColumns (const vector<unsigned> & cols)
{
  inter_cols = cols;
  for (unsigned ii = 0; ii < inter_cols.size(); ++ii){
    if (inter_cols[ii] == 0){
      cerr << "invalid col index, should be larger than 0" << endl;
      throw;
    }
    inter_cols[ii] -= 1;
  }
}

void 
TableFileLoader::
setEvery (const unsigned every_) 
{
  every = every_;
}


bool
TableFileLoader::
loadLine (vector<double > & odata)
{
  char valueline [MaxLineLength];
  
  while (data.getline(valueline, MaxLineLength)){
    if (valueline[0] == '#' || valueline[0] == '@'){
      continue;
    }
    else if (count_read++ % every == 0){
      break;
    }
  }
  
  if (data.eof()){
    return false;
  }
  else if (! data.good()){
    cerr << "error file reading state!" << endl;
    throw;
  }

  vector<string > words;
  StringOperation::split (string(valueline), words);
  odata.resize (inter_cols.size());

  for (unsigned ii = 0; ii < inter_cols.size(); ++ii){
    odata[ii] = atof(words[inter_cols[ii]].c_str());
  }
  
  return true;
}

void
TableFileLoader::
loadAll (vector<vector<double > > & odata)
{
  odata.resize(inter_cols.size());
  vector<double > line;
  while (loadLine (line)){
    for (unsigned ii = 0; ii < inter_cols.size(); ++ii){
      odata[ii].push_back (line[ii]);
    }
  }
}


