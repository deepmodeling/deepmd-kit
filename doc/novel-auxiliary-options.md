# Novel Auxiliary Options
## Type embedding
Instead of training embedding net for each atom pair (regard as G_ij, and turns out to be N^2 networks), we now share a public embedding net (regard as G) and present each atom with a special vector, named as type embedding (v_i). So, our algorithm for generating a description change from G_ij(s_ij) to G(s_ij, v_i, v_j).
1. We obtain the type embedding by a small embedding net, projecting atom type to embedding vector.
2. As for the fitting net, we fix the type embedding and replace individual fitting net with shared fitting net. (while adding type embedding information to its input)

### Training hyper-parameter
descriptor:  
"type" : "se_a_ebd"  # for applying share embedding algorithm  
"type_filter" : list # network architecture of the small embedding net, which output type embedding  
"type_one_side" : bool  # when generating descriptor, whether use the centric atom type embedding (true: G(s_ij, v_i, v_j), false: G(s_ij, v_j))  
  
fitting_net:  
"share_fitting" : bool # if applying share fitting net, set true  


## Interpolation with tabulated pair potentials
