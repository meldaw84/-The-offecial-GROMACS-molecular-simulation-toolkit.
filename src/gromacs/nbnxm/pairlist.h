/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2012- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */

#ifndef GMX_NBNXM_PAIRLIST_H
#define GMX_NBNXM_PAIRLIST_H

#include <cstddef>

#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdtypes/locality.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/defaultinitializationallocator.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/real.h"

#include "pairlistparams.h"

struct nbnxn_atomdata_t;
struct NbnxnPairlistCpuWork;
struct NbnxnPairlistGpuWork;
struct t_nblist;
enum class ClusterDistanceKernelType : int;

namespace gmx
{
template<typename T>
class ListOfLists;
}

namespace Nbnxm
{
class Grid;
class GridSet;
} // namespace Nbnxm

//! Convenience type for vector with aligned memory
template<typename T>
using AlignedVector = std::vector<T, gmx::AlignedAllocator<T>>;

//! Convenience type for vector that avoids initialization at resize()
template<typename T>
using FastVector = std::vector<T, gmx::DefaultInitializationAllocator<T>>;

/*! \brief Cache-line protection buffer
 *
 * A buffer data structure of 64 bytes
 * to be placed at the beginning and end of structs
 * to avoid cache invalidation of the real contents
 * of the struct by writes to neighboring memory.
 */
typedef struct
{
    //! Unused field used to create space to protect cache lines that are in use
    int dummy[16];
} gmx_cache_protect_t;

/*! \brief This is the actual cluster-pair list j-entry.
 *
 * cj is the j-cluster.
 * The interaction bits in excl are indexed i-major, j-minor.
 * The cj entries are sorted such that ones with exclusions come first.
 * This means that once a full mask (=NBNXN_INTERACTION_MASK_ALL)
 * is found, all subsequent j-entries in the i-entry also have full masks.
 */
struct nbnxn_cj_t
{
    //! The j-cluster
    int cj;
    //! The exclusion (interaction) bits
    unsigned int excl;
};

//! Simple j-cluster list
class JClusterList
{
public:
    //! The list of packed j-cluster groups
    FastVector<nbnxn_cj_t> list_;
    //! Return the j-cluster index for \c index from the pack list
    int cj(int index) const { return list_[index].cj; }
    //! Return the exclusion mask for \c index
    const unsigned int& excl(int index) const { return list_[index].excl; }
    //! Return the exclusion mask for \c index
    unsigned int& excl(int index) { return list_[index].excl; }
    //! Return the size of the list (not the number of packed elements)
    gmx::index size() const noexcept { return list_.size(); }
    //! Return whether the list is empty
    bool empty() const noexcept { return size() == 0; }
    //! Resize the list
    void resize(gmx::index count) { list_.resize(count); }
    //! Add a new element to the list
    void push_back(const decltype(list_)::value_type& value) { list_.push_back(value); }
};

/*! \brief Constants for interpreting interaction flags
 *
 * In nbnxn_ci_t the integer shift contains the shift in the lower 7 bits.
 * The upper bits contain information for non-bonded kernel optimization.
 * Simply calculating LJ and Coulomb for all pairs in a cluster pair is fine.
 * But three flags can be used to skip interactions, currently only for subc=0
 * !(shift & NBNXN_CI_DO_LJ(subc))   => we can skip LJ for all pairs
 * shift & NBNXN_CI_HALF_LJ(subc)    => we can skip LJ for the second half of i
 * !(shift & NBNXN_CI_DO_COUL(subc)) => we can skip Coulomb for all pairs
 */
//! \{
#define NBNXN_CI_SHIFT 127
#define NBNXN_CI_DO_LJ(subc) (1 << (7 + 3 * (subc)))
#define NBNXN_CI_HALF_LJ(subc) (1 << (8 + 3 * (subc)))
#define NBNXN_CI_DO_COUL(subc) (1 << (9 + 3 * (subc)))
//! \}

/*! \brief Cluster-pair Interaction masks
 *
 * Bit i*j-cluster-size + j tells if atom i and j interact.
 */
//! \{
// TODO: Rename according to convention when moving into Nbnxn namespace
//! All interaction mask is the same for all kernels
constexpr unsigned int NBNXN_INTERACTION_MASK_ALL = 0xffffffffU;
//! 4x4 kernel diagonal mask
constexpr unsigned int NBNXN_INTERACTION_MASK_DIAG = 0x08ceU;
//! 4x2 kernel diagonal masks
//! \{
constexpr unsigned int NBNXN_INTERACTION_MASK_DIAG_J2_0 = 0x0002U;
constexpr unsigned int NBNXN_INTERACTION_MASK_DIAG_J2_1 = 0x002fU;
//! \}
//! 4x8 kernel diagonal masks
//! \{
constexpr unsigned int NBNXN_INTERACTION_MASK_DIAG_J8_0 = 0xf0f8fcfeU;
constexpr unsigned int NBNXN_INTERACTION_MASK_DIAG_J8_1 = 0x0080c0e0U;
//! \}
//! \}

/*! \brief Lower limit for square interaction distances in nonbonded kernels.
 *
 * For smaller values we will overflow when calculating r^-1 or r^-12, but
 * to keep it simple we always apply the limit from the tougher r^-12 condition.
 */
#if GMX_DOUBLE
// Some double precision SIMD architectures use single precision in the first
// step, so although the double precision criterion would allow smaller rsq,
// we need to stay in single precision with some margin for the N-R iterations.
constexpr double c_nbnxnMinDistanceSquared = 1.0e-36;
#else
// The worst intermediate value we might evaluate is r^-12, which
// means we should ensure r^2 stays above pow(GMX_FLOAT_MAX,-1.0/6.0)*1.01 (some margin)
constexpr float c_nbnxnMinDistanceSquared = 3.82e-07F; // r > 6.2e-4
#endif


//! The number of clusters in a super-cluster, used for GPU
constexpr int c_nbnxnGpuNumClusterPerSupercluster = 8;
static_assert(c_nbnxnGpuNumClusterPerSupercluster
                      == c_gpuNumClusterPerCellX * c_gpuNumClusterPerCellY * c_gpuNumClusterPerCellZ,
              "c_nbnxnGpuNumClusterPerSupercluster needs to match the number of clusters per "
              "search cell");

/*! \brief With GPU kernels we group cluster pairs in 4 to optimize memory usage
 * of integers containing 32 bits.
 */
constexpr int c_nbnxnGpuJgroupSize = (32 / c_nbnxnGpuNumClusterPerSupercluster);

/*! \internal
 * \brief Simple pair-list i-unit
 */
struct nbnxn_ci_t
{
    //! i-cluster
    int ci;
    //! Shift vector index plus possible flags, see above
    int shift;
    //! Start index into cj
    int cj_ind_start;
    //! End index into cj
    int cj_ind_end;
};

//! Grouped pair-list i-unit
typedef struct nbnxn_sci
{
    //! Returns the number of j-cluster groups in this entry
    int numJClusterGroups() const { return cjPackedEnd - cjPackedBegin; }

    //! i-super-cluster
    int sci;
    //! Shift vector index plus possible flags
    int shift;
    //! Start index into cjPacked
    int cjPackedBegin;
    //! End index into cjPacked (ie. one past the last element)
    int cjPackedEnd;
} nbnxn_sci_t;

//! Interaction data for a j-group for one warp
struct nbnxn_im_ei_t
{
    //! The i-cluster interactions mask for 1 warp
    unsigned int imask = 0U;
    //! Index into the exclusion array for 1 warp, default index 0 which means no exclusions
    int excl_ind = 0;
};

//! Packed j-cluster list element
typedef struct
{
    //! The packed j-clusters
    int cj[c_nbnxnGpuJgroupSize];
    //! The i-cluster mask data for 2 warps
    nbnxn_im_ei_t imei[c_nbnxnGpuClusterpairSplit];
} nbnxn_cj_packed_t;

//! Return the index of an j-atom within a warp
constexpr int jParticleIndexWithinWarp(gmx::index index)
{
    return index & (c_nbnxnGpuClusterSize / c_nbnxnGpuClusterpairSplit - 1);
}

/*! Packed j-cluster list
 *
 * Four j-cluster indices are stored per integer in an nbnxn_cj_packed_t.
 */
class PackedJClusterList
{
public:
    explicit PackedJClusterList(const gmx::PinningPolicy pinningPolicy) :
        list_({}, { pinningPolicy })
    {
    }
    //! The list of packed j-cluster groups
    gmx::HostVector<nbnxn_cj_packed_t> list_;
    //! Return the index of particle \c index within its group
    static constexpr gmx::index indexOfParticleWithinGroup(gmx::index index)
    {
        return index & (c_nbnxnGpuJgroupSize - 1);
    }
    //! Convert a j-cluster index to a cjPacked group index
    static constexpr gmx::index clusterIndexToGroupIndex(gmx::index jClusterIndex)
    {
        return jClusterIndex / c_nbnxnGpuJgroupSize;
    }

    //! Return the j-cluster index for \c index from the pack list
    int cj(const int index) const
    {
        return list_[index / c_nbnxnGpuJgroupSize].cj[indexOfParticleWithinGroup(index)];
    }
    //! Return the i-cluster interaction mask for the first cluster in \c index
    unsigned int imask0(const int index) const
    {
        return list_[index / c_nbnxnGpuJgroupSize].imei[0].imask;
    }
    //! Return the size of the list (not the number of packed elements)
    gmx::index size() const noexcept { return list_.size(); }
    //! Return whether the list is empty
    bool empty() const noexcept { return size() == 0; }
    //! Resize the packed list
    void resize(gmx::index count) { list_.resize(count); }
    //! Add a new element to the packed list
    void push_back(const decltype(list_)::value_type& value) { list_.push_back(value); }
    static_assert(sizeof(list_[0].imei[0].imask) * 8 >= c_nbnxnGpuJgroupSize * c_gpuNumClusterPerCell,
                  "The i super-cluster cluster interaction mask does not contain a sufficient "
                  "number of bits");
};

//! Struct for storing the atom-pair interaction bits for a cluster pair in a GPU pairlist
struct nbnxn_excl_t
{
    //! Constructor, sets no exclusions, so all atom pairs interacting
    MSVC_DIAGNOSTIC_IGNORE(26495) // pair is not being initialized!
    nbnxn_excl_t()
    {
        for (unsigned int& pairEntry : pair)
        {
            pairEntry = NBNXN_INTERACTION_MASK_ALL;
        }
    }
    MSVC_DIAGNOSTIC_RESET

    //! Topology exclusion interaction bits per warp
    unsigned int pair[c_nbnxnGpuExclSize];
};

//! Cluster pairlist type for use on CPUs
struct NbnxnPairlistCpu
{
    NbnxnPairlistCpu();

    //! Print statistics of a pair list, used for debug output
    void printNblistStatistics(FILE* fp, const Nbnxm::GridSet& gridSet, const real rl) const;
    //! Makes the cluster list for each grid cell from \c firstCell to \c lastCell
    void makeClusterListDispatcher(const Nbnxm::Grid&              iGrid,
                                   int                             ci,
                                   const Nbnxm::Grid&              jGrid,
                                   int                             firstCell,
                                   int                             lastCell,
                                   bool                            excludeSubDiagonal,
                                   const nbnxn_atomdata_t*         nbat,
                                   const real                      rlist2,
                                   const real                      rbb2,
                                   const ClusterDistanceKernelType kernelType,
                                   int*                            numDistanceChecks);
    /*! \brief Make a pair list for the perturbed pairs, while excluding
     * them from the Verlet list.
     *
     * This is only done to avoid singularities for overlapping particles
     * (from 0/0), since the charges and LJ parameters have been zeroed in
     * the nbnxn data structure. */
    void makeFepList(gmx::ArrayRef<const int> atomIndices,
                     const nbnxn_atomdata_t*  nbat,
                     gmx_bool                 bDiagRemoved,
                     real gmx_unused          shx,
                     real gmx_unused          shy,
                     real gmx_unused          shz,
                     real gmx_unused          rlist_fep2,
                     const Nbnxm::Grid&       iGrid,
                     const Nbnxm::Grid&       jGrid,
                     t_nblist*                nlist);
    //! Make a new ci entry at the back
    void addNewIEntry(int ciIndex, int shift, int flags);
    //! Close this simple list i entry
    void closeIEntry(int gmx_unused      sp_max_av,
                     gmx_bool gmx_unused progBal,
                     float gmx_unused    nsp_tot_est,
                     int gmx_unused      thread,
                     int gmx_unused      nthread);
    //! Dummy function so this class works like NbnxmPairlistGpu
    void syncWork() const;
    //! Clears pairlists
    void clear();
    //! Debug list print function
    void printNblist(FILE* fp);
    //! Return whether the pairlist is simple (ie. not for a GPU)
    static bool isSimple() { return true; }
    /*! \brief SIMD code for checking and adding cluster-pairs to the list
     * using coordinates in packed format.
     *
     * Checks bounding box distances and possibly atom pair distances.
     *
     * Three flavours are implemented that make cluster lists that suit
     * respectively the plain-C, SIMD 4xn, and SIMD 4xnn kernel flavours.
     *
     * \param[in]     jGrid               The j-grid
     * \param[in]     icluster            The index of the i-cluster
     * \param[in]     jclusterFirst       The first cluster in the j-range, using i-cluster size indexing
     * \param[in]     jclusterLast        The last cluster in the j-range, using i-cluster size indexing
     * \param[in]     excludeSubDiagonal  Exclude atom pairs with i-index > j-index
     * \param[in]     x_j                 Coordinates for the j-atom, in SIMD packed format
     * \param[in]     rlist2              The squared list cut-off
     * \param[in]     rbb2                The squared cut-off for putting cluster-pairs in the list based on bounding box distance only
     * \param[in,out] numDistanceChecks   The number of distance checks performed
     */
    //! \{
    void makeClusterListPlainC(const Nbnxm::Grid&       jGrid,
                               int                      icluster,
                               int                      jclusterFirst,
                               int                      jclusterLast,
                               bool                     excludeSubDiagonal,
                               const real* gmx_restrict x_j,
                               real                     rlist2,
                               float                    rbb2,
                               int* gmx_restrict        numDistanceChecks);
    void makeClusterListSimd4xn(const Nbnxm::Grid&       jGrid,
                                int                      icluster,
                                int                      firstCell,
                                int                      lastCell,
                                bool                     excludeSubDiagonal,
                                const real* gmx_restrict x_j,
                                real                     rlist2,
                                float                    rbb2,
                                int* gmx_restrict        numDistanceChecks);
    void makeClusterListSimd2xnn(const Nbnxm::Grid&       jGrid,
                                 int                      icluster,
                                 int                      firstCell,
                                 int                      lastCell,
                                 bool                     excludeSubDiagonal,
                                 const real* gmx_restrict x_j,
                                 real                     rlist2,
                                 float                    rbb2,
                                 int* gmx_restrict        numDistanceChecks);
    //! \}
    //! Return the number of simple j clusters in this list
    int getNumSimpleJClustersInList() const { return cj.size(); }
    //! Increment the number of simple j clusters in this list
    void incrementNumSimpleJClustersInList(int ncj_old_j);
    /*! \brief Set all atom-pair exclusions for the last i-cluster entry
     * in the CPU list.
     *
     * All the atom-pair exclusions from the topology are
     * converted to exclusion masks in the simple pairlist. */
    void setExclusionsForIEntry(const Nbnxm::GridSet&        gridSet,
                                gmx_bool                     diagRemoved,
                                int gmx_unused               na_cj_2log,
                                const gmx::ListOfLists<int>& exclusions);

    //! Cache protection
    gmx_cache_protect_t cp0;

    //! The number of atoms per i-cluster
    int na_ci;
    //! The number of atoms per j-cluster
    int na_cj;
    //! The radius for constructing the list
    real rlist;
    //! The i-cluster list
    FastVector<nbnxn_ci_t> ci;
    //! The outer, unpruned i-cluster list
    FastVector<nbnxn_ci_t> ciOuter;

    //! The j-cluster list
    JClusterList cj;
    //! The outer, unpruned j-cluster list
    FastVector<nbnxn_cj_t> cjOuter;
    //! The number of j-clusters that are used by ci entries in this list, will be <= cj.list.size()
    int ncjInUse;

    //! Working data storage for list construction
    std::unique_ptr<NbnxnPairlistCpuWork> work;

    //! Cache protection
    gmx_cache_protect_t cp1;
};

/* Cluster pairlist type, with extra hierarchies, for on the GPU
 *
 * NOTE: for better performance when combining lists over threads,
 *       all vectors should use default initialization. But when
 *       changing this, excl should be initialized when adding entries.
 */
struct NbnxnPairlistGpu
{
    /*! \brief Constructor
     *
     * \param[in] pinningPolicy  Sets the pinning policy for all buffers used on the GPU
     */
    NbnxnPairlistGpu(gmx::PinningPolicy pinningPolicy);

    //! Print statistics of a pair list, used for debug output
    void printNblistStatistics(FILE* fp, const Nbnxm::GridSet& gridSet, const real rl) const;
    /*! \brief Returns a reference to the exclusion mask for
     * j-cluster group \p cjPackedIndex and warp \p warp
     *
     * Generates a new exclusion entry when the j-cluster group
     * uses the default all-interaction mask at call time, so the
     * returned mask can be modified when needed. */
    nbnxn_excl_t& getExclusionMask(int cjPackedIndex, int warp);

    /*! \brief Sets self exclusions and excludes half of the double pairs in the self cluster-pair \p cjPacked.list_[cjPackedIndex].cj[jOffsetInGroup]
     *
     * \param[in]     cjPackedIndex   The j-cluster group index into \p cjPacked
     * \param[in]     jOffsetInGroup  The j-entry offset in \p cjPacked.list_[cjPackedIndex]
     * \param[in]     iClusterInCell  The i-cluster index in the cell
     */
    void setSelfAndNewtonExclusionsGpu(int cjPackedIndex, int jOffsetInGroup, int iClusterInCell);

    /*! \brief Makes a pair list of super-cell sci vs scj.
     *
     * Checks bounding box distances and possibly atom pair distances.
     *
     * Has both a SIMD4 implementation (if supported) and a plain C
     * fallback implementation. */
    void makeClusterListSupersub(const Nbnxm::Grid& iGrid,
                                 const Nbnxm::Grid& jGrid,
                                 int                sci,
                                 int                scj,
                                 bool               excludeSubDiagonal,
                                 int                stride,
                                 const real*        x,
                                 const real         rlist2,
                                 float              rbb2,
                                 int*               numDistanceChecks);

    //! Makes the cluster list for each grid cell from \c firstCell to \c lastCell
    void makeClusterListDispatcher(const Nbnxm::Grid&        iGrid,
                                   int                       ci,
                                   const Nbnxm::Grid&        jGrid,
                                   int                       firstCell,
                                   int                       lastCell,
                                   bool                      excludeSubDiagonal,
                                   const nbnxn_atomdata_t*   nbat,
                                   const real                rlist2,
                                   const real                rbb2,
                                   ClusterDistanceKernelType kernelType,
                                   int*                      numDistanceChecks);
    /*! \brief Make a pair list for the perturbed pairs, while excluding
     * them from the Verlet list.
     *
     * This is only done to avoid singularities for overlapping particles
     * (from 0/0), since the charges and LJ parameters have been zeroed in
     * the nbnxn data structure. */
    void makeFepList(gmx::ArrayRef<const int> atomIndices,
                     const nbnxn_atomdata_t*  nbat,
                     gmx_bool                 bDiagRemoved,
                     real gmx_unused          shx,
                     real gmx_unused          shy,
                     real gmx_unused          shz,
                     real gmx_unused          rlist_fep2,
                     const Nbnxm::Grid&       iGrid,
                     const Nbnxm::Grid&       jGrid,
                     t_nblist*                nlist);
    /*! \brief Set all atom-pair exclusions for the last i-super-cluster
     * entry in the GPU list
     *
     * All the atom-pair exclusions from the topology are
     * converted to exclusion masks in the simple pairlist. */
    void setExclusionsForIEntry(const Nbnxm::GridSet&        gridSet,
                                gmx_bool                     diagRemoved,
                                int gmx_unused               na_cj_2log,
                                const gmx::ListOfLists<int>& exclusions);
    //! Make a new sci entry at the back
    void addNewIEntry(int sciIndex, int shift, int /* flags */);
    /*! \brief Split sci entry for load balancing on the GPU.
     *
     * Splitting ensures we have enough lists to fully utilize the whole GPU.
     * With progBal we generate progressively smaller lists, which improves
     * load balancing. As we only know the current count on our own thread,
     * we will need to estimate the current total amount of i-entries.
     * As the lists get concatenated later, this estimate depends
     * both on nthread and our own thread index. */
    void splitSciEntry(int nsp_target_av, gmx_bool progBal, float nsp_tot_est, int thread, int nthread);
    //! Close this super/sub list i entry
    void closeIEntry(int nsp_max_av, gmx_bool progBal, float nsp_tot_est, int thread, int nthread);
    //! Syncs the working array before adding another grid pair to the GPU list
    void syncWork() const;
    //! Clears pairlists
    void clear();
    //! Debug list print function
    void printNblist(FILE* fp);
    //! Return whether the pairlist is simple (ie. not for a GPU)
    static bool isSimple() { return false; }
    //! Return the number of simple j clusters in this list (ie. 0 for this GPU list)
    static int getNumSimpleJClustersInList() { return 0; }
    //! Empty function because a GPU pairlist does not use simple j clusters.
    void incrementNumSimpleJClustersInList(int /* numJClusters */);

    //! Cache protection
    gmx_cache_protect_t cp0;

    //! The number of atoms per i-cluster
    int na_ci;
    //! The number of atoms per j-cluster
    int na_cj;
    //! The number of atoms per super cluster
    int na_sc;
    //! The radius for constructing the list
    real rlist;
    //! The i-super-cluster list, indexes into cjPacked list;
    gmx::HostVector<nbnxn_sci_t> sci;
    //! The list of packed j-cluster groups
    PackedJClusterList cjPacked;
    //! Atom interaction bits (non-exclusions)
    gmx::HostVector<nbnxn_excl_t> excl;
    //! The total number of i-clusters
    int nci_tot;

    //! Working data storage for list construction
    std::unique_ptr<NbnxnPairlistGpuWork> work;

    //! Cache protection
    gmx_cache_protect_t cp1;
};

#endif
