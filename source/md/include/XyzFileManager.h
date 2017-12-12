#ifndef __XyzFileManager_h_wanghan__
#define __XyzFileManager_h_wanghan__

#include <vector>
using namespace std;

namespace XyzFileManager{

  void
  read (const string & file,
	vector<string > & atom_name,
	vector<vector<double > > & posi,
	vector<vector<double > > & velo,
	vector<vector<double > > & forc,
	vector<double > & boxsize);

  void
  getBoxSize (const string & name,
	      vector<double > & boxsize);

};

#endif
