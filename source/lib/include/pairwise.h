// SPDX-License-Identifier: LGPL-3.0-or-later
#include <vector>

namespace deepmd {
/**
 * Group atoms into different fragments according to indexes.
 *
 * @param[out] fragments The indexes of atoms that each fragment contains.
 * Fragment has been sorted.
 * @param[in] idxs The indexes of the fragment that each atom belongs to. -1
 * will be ignored.
 */
void group_atoms_cpu(std::vector<std::vector<int>> &fragments,
                     const std::vector<int> &idxs);
/**
 * DPRc pairwise map.
 *
 * @param[out] forward_qm_map Forward map for QM atoms.
 * @param[out] backward_qm_map Backward map for QM atoms.
 * @param[out] forward_qmmm_map Forward map for QM/MM atoms.
 * @param[out] backward_qmmm_map Backward map for QM/MM atoms.
 * @param[out] nloc_qm The number of local QM atoms.
 * @param[out] nloc_qmmm The number of local QM/MM atoms.
 * @param[out] nall_qm The number of all QM atoms, including local and ghost
 * atoms.
 * @param[out] nall_qmmm The number of all QM/MM atoms, including local and
 * ghost atoms.
 * @param[in] fragments The indexes of atoms that each fragment contains.
 * Assume that only the first fragment consists of QM atoms.
 * @param[in] nloc The number of local atoms.
 * @param[in] nall The number of all atoms, including local and ghost atoms.
 */
void dprc_pairwise_map_cpu(std::vector<int> &forward_qm_map,
                           std::vector<int> &backward_qm_map,
                           std::vector<int> &forward_qmmm_map,
                           std::vector<int> &backward_qmmm_map,
                           int &nloc_qm,
                           int &nloc_qmmm,
                           int &nall_qm,
                           int &nall_qmmm,
                           const std::vector<std::vector<int>> &fragments,
                           const int nloc,
                           const int nall);
}  // namespace deepmd
