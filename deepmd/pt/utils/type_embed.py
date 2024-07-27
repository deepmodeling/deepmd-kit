import torch

def embed_atom_type(
    ntypes: int,
    natoms: torch.Tensor,
    type_embedding: torch.Tensor,
):
    """Make the embedded type for the atoms in system.
    The atoms are assumed to be sorted according to the type,
    thus their types are described by a `torch.Tensor` natoms, see explanation below.

    Parameters
    ----------
    ntypes:
        Number of types.
    natoms:
        The number of atoms. This tensor has the length of Ntypes + 2
        natoms[0]: number of local atoms
        natoms[1]: total number of atoms held by this processor
        natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
    type_embedding:
        The type embedding.
        It has the shape of [ntypes, embedding_dim]

    Returns
    -------
    atom_embedding
        The embedded type of each atom.
        It has the shape of [numb_atoms, embedding_dim]
    """
    te_out_dim = type_embedding.size(-1)
    atype = []
    for ii in range(ntypes):
        atype.append(torch.full((natoms[2 + ii],), ii, dtype=torch.int32))
    atype = torch.cat(atype, dim=0)
    atm_embed = type_embedding[atype.long()]  # (nf*natom)*nchnl
    atm_embed = atm_embed.view(-1, te_out_dim)
    return atm_embed

