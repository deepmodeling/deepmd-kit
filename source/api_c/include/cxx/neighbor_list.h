#pragma once
#include <stdio.h>

namespace deepmd
{

    /**
     * @brief             Construct InputNlist with the input LAMMPS nbor list info.
     *
     * @struct            InputNlist
     */
    struct InputNlist
    {
        /// Number of core region atoms
        int inum;
        /// Array stores the core region atom's index
        int *ilist;
        /// Array stores the core region atom's neighbor atom number
        int *numneigh;
        /// Array stores the core region atom's neighbor index
        int **firstneigh;
        InputNlist()
            : inum(0), ilist(NULL), numneigh(NULL), firstneigh(NULL){};
        InputNlist(
            int inum_,
            int *ilist_,
            int *numneigh_,
            int **firstneigh_)
            : inum(inum_), ilist(ilist_), numneigh(numneigh_), firstneigh(firstneigh_){};
        ~InputNlist(){};
    };
}