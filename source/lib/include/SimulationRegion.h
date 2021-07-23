#ifndef __SimulationRegion_h_wanghan__
#define __SimulationRegion_h_wanghan__

#define MOASPNDIM 3
#include "utilities.h"
#include <fstream>

  template<typename VALUETYPE>
  class SimulationRegion
  {
protected:
    const static int SPACENDIM = MOASPNDIM;
public:
    void reinitBox (const double * boxv);
    void affineTransform (const double * affine_map);
    void reinitOrigin (const double * orig);
    void reinitOrigin (const std::vector<double> & orig);
    void backup  ();
    void recover ();
public:
    SimulationRegion ();
    ~SimulationRegion ();
    double *		getBoxTensor	()		{return boxt;};
    const double *	getBoxTensor	() const	{return boxt;};
    double *		getRecBoxTensor ()		{return rec_boxt;}
    const double *	getRecBoxTensor () const	{return rec_boxt;}
    double *		getBoxOrigin	()		{return origin;}
    const double *	getBoxOrigin	() const	{return origin;}
    double		getVolume	() const	{return volume;}
public:
    void		toFaceDistance	(double * dd) const;
public:
    void phys2Inter (double * i_v, const VALUETYPE * p_v) const;
    void inter2Phys (VALUETYPE * p_v, const double * i_v) const;
public:
    bool		isPeriodic	(const int dim) const {return is_periodic[dim];}
    static int		compactIndex	(const int * idx) ;
    double *		getShiftVec	(const int index = 0) ;
    const double *	getShiftVec	(const int index = 0) const;
    int			getShiftIndex	(const int * idx) const;
    int			getNullShiftIndex() const;
    void		shiftCoord	(const int * idx,
					 VALUETYPE &x,
					 VALUETYPE &y,
					 VALUETYPE &z) const;
    static int		getNumbShiftVec ()	 {return shift_info_size;}
    static int		getShiftVecTotalSize ()  {return shift_vec_size;}
public:
    void 
    diffNearestNeighbor (const VALUETYPE * r0,
			 const VALUETYPE * r1,
			 VALUETYPE * phys) const;
    virtual void 
    diffNearestNeighbor (const VALUETYPE x0,
			 const VALUETYPE y0,
			 const VALUETYPE z0,
			 const VALUETYPE x1,
			 const VALUETYPE y1,
			 const VALUETYPE z1,
			 VALUETYPE & dx,
			 VALUETYPE & dy,
			 VALUETYPE & dz) const ;
    virtual void diffNearestNeighbor (const VALUETYPE x0,
				      const VALUETYPE y0,
				      const VALUETYPE z0,
				      const VALUETYPE x1,
				      const VALUETYPE y1,
				      const VALUETYPE z1,
				      VALUETYPE & dx,
				      VALUETYPE & dy,
				      VALUETYPE & dz,
				      int & shift_x,
				      int & shift_y,
				      int & shift_z) const ;
    virtual void diffNearestNeighbor (const VALUETYPE x0,
				      const VALUETYPE y0,
				      const VALUETYPE z0,
				      const VALUETYPE x1,
				      const VALUETYPE y1,
				      const VALUETYPE z1,
				      VALUETYPE & dx,
				      VALUETYPE & dy,
				      VALUETYPE & dz,
				      VALUETYPE & shift_x,
				      VALUETYPE & shift_y,
				      VALUETYPE & shift_z) const ;
private:
    void computeVolume ();
    void computeRecBox ();
    double		volume;
    double		volumei;
    double		boxt		[SPACENDIM*SPACENDIM];
    double		boxt_bk		[SPACENDIM*SPACENDIM];
    double		rec_boxt	[SPACENDIM*SPACENDIM];
    double		origin		[SPACENDIM];
    bool		is_periodic	[SPACENDIM];
    std::string		class_name;
    bool		enable_restart;
protected:
    void computeShiftVec ();
    const static int			DBOX_XX = 1;
    const static int			DBOX_YY = 1;
    const static int			DBOX_ZZ = 1;
    const static int			NBOX_XX = DBOX_XX*2+1;
    const static int			NBOX_YY = DBOX_YY*2+1;
    const static int			NBOX_ZZ = DBOX_ZZ*2+1;
    const static int			shift_info_size = NBOX_XX * NBOX_YY * NBOX_ZZ;
    const static int			shift_vec_size = SPACENDIM * shift_info_size;
    double				shift_vec	[shift_vec_size];
    double				inter_shift_vec [shift_vec_size];
    static int index3to1 (const int tx, const int ty, const int tz) 
	{
	  return (NBOX_ZZ * (NBOX_YY * (tx+DBOX_XX) + ty+DBOX_YY)+ tz+DBOX_ZZ);
	}    
    double *		getInterShiftVec	(const int index = 0) ;
    const double *	getInterShiftVec	(const int index = 0) const;
private:
    void copy	    (double * o_v, const double * i_v) const;
    void naiveTensorDotVector (double * out,
			       const double * i_t,
			       const double * i_v) const;
    void naiveTensorTransDotVector (double * out,
				    const double * i_t,
				    const double * i_v) const;
    void tensorDotVector (double * out,
			  const double * i_t,
			  const double * i_v) const;
    void tensorTransDotVector (double * out,
			       const double * i_t,
			       const double * i_v) const;
    void getFromRestart (double * my_boxv, double * my_orig, bool * period) const;
    void defaultInitBox (double * my_boxv, double * my_orig, bool * period) const;
    void apply_periodic (int dim, double * dd) const;
    void apply_periodic (int dim, double * dd, int & shift) const;
private:
    std::fstream fp;
  };

#ifdef MOASP_INLINE_IMPLEMENTATION
#include "SimulationRegion_Impl.h"
#endif

#endif


