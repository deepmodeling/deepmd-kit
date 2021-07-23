#ifndef __XyzFileManager_h_wanghan__
#define __XyzFileManager_h_wanghan__

#include <vector>
// using namespace std;

namespace XyzFileManager{

  void
  read (const std::string & file,
	std::vector<std::string > & atom_name,
	std::vector<std::vector<double > > & posi,
	std::vector<std::vector<double > > & velo,
	std::vector<std::vector<double > > & forc);
  
};

#endif
