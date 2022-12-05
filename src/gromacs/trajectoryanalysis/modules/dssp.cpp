/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2022- The GROMACS Authors
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
/*! \internal \file
 * \brief
 * Implements gmx::analysismodules::Dssp.
 *
 * \author Sergey Gorelov <gorelov_sv@pnpi.nrcki.ru>
 * \author Anatoly Titov <titov_ai@pnpi.nrcki.ru>
 * \author Alexey Shvetsov <alexxyum@gmail.com>
 * \ingroup module_trajectoryanalysis
 */

#include "dssp.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>

#include "gromacs/fileio/gmxfio.h"
#include "gromacs/math/units.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/selection/nbsearch.h"
#include "gromacs/selection/selectionoption.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/topologyinformation.h"
#include "gromacs/utility/fatalerror.h"

namespace gmx
{

namespace analysismodules
{

namespace
{

//! Structure that contains storage information from different frames.
struct DsspStorageFrame
{
    //! Frame number.
    int frameNumber_ = 0;
    //! Frame dssp data.
    std::string dsspData_;
};

/*! \brief
 * Class that stores frame information in storage and, upon request, can return it.
 */
class DsspStorage
{
public:
    /*! \brief
     * Function that storeges frame information in storage.
     */
    void addData(int frnr, const std::string& data);
    /*! \brief
     * Function that returns frame information from storage.
     */
    std::vector<DsspStorageFrame> const& getData();

private:
    /*! \brief
     * Vector that contains information from different frames.
     */
    std::vector<DsspStorageFrame> data_;
};

void DsspStorage::addData(int frnr, const std::string& data)
{
    DsspStorageFrame dsspData_;
    dsspData_.frameNumber_ = frnr;
    dsspData_.dsspData_    = data;
    data_.push_back(dsspData_);
}

std::vector<DsspStorageFrame> const& DsspStorage::getData()
{
    return data_;
}

//! Enum of backbone atoms' types.
enum class BackboneAtomTypes : std::size_t
{
    AtomCA,
    AtomC,
    AtomO,
    AtomN,
    AtomH,
    Count
};

//! String values corresponding to backbone atom types.
const gmx::EnumerationArray<BackboneAtomTypes, const char*> c_backboneAtomTypeNames = {
    { "CA", "C", "O", "N", "H" }
};

/*! \brief
 * Structure of residues' information that can operate with atoms' indices.
 */
struct ResInfo
{
    /*! \brief
     * Size_t array of atoms' indices corresponding to backbone atom types.
     */
    std::array<std::size_t, static_cast<std::size_t>(BackboneAtomTypes::Count)> backboneIndices_ = { 0, 0, 0, 0, 0 };
    /*! \brief
     * Bool array determining whether backbone atoms have been assigned.
     */
    std::array<bool, static_cast<std::size_t>(BackboneAtomTypes::Count)>
            backboneIndicesStatus_ = { false, false, false, false, false };
    /*! \brief
     * Function that returns atom's index based on specific atom type.
     */
    std::size_t getIndex(BackboneAtomTypes atomTypeName) const;
    //! Pointer to t_resinfo which contains full information about this specific residue.
    t_resinfo* info_ = nullptr;
    //! Pointer to t_resinfo which contains full information about this residue's h-bond donors.
    t_resinfo* donor_[2] = { nullptr, nullptr };
    //! Pointer to t_resinfo which contains full information about this residue's h-bond acceptors.
    t_resinfo* acceptor_[2] = { nullptr, nullptr };
    //! Pointer to previous residue in list.
    ResInfo* prevResi_ = nullptr;
    //! Pointer to next residue in list.
    ResInfo* nextResi_ = nullptr;
    //! Float value of h-bond energy with this residue's donors.
    float donorEnergy_[2] = { 0, 0 };
    //! Float value of h-bond energy with this residue's accpetors.
    float acceptorEnergy_[2] = { 0, 0 };
    //! Bool value that defines either this residue is proline (PRO) or not.
    bool isProline_ = false;
};

std::size_t ResInfo::getIndex(BackboneAtomTypes atomTypeName) const
{
    return backboneIndices_[static_cast<std::size_t>(atomTypeName)];
}

//! Enum of secondary structures' types.
enum class SecondaryStructureTypes : std::size_t
{
    Loop = 0, //! ~
    Break,    //! =
    Bend,     //! S
    Turn,     //! T
    Helix_PP, //! P
    Helix_5,  //! I
    Helix_3,  //! G
    Strand,   //! E
    Bridge,   //! B
    Helix_4,  //! H
    Count

};

//! String values corresponding to secondary structures' types.
const gmx::EnumerationArray<SecondaryStructureTypes, const char> c_secondaryStructureTypeNames = {
    { '~', '=', 'S', 'T', 'P', 'I', 'G', 'E', 'B', 'H' }
};

//! Enum of turns' types.
enum class TurnsTypes : std::size_t
{
    Turn_3 = 0,
    Turn_4,
    Turn_5,
    Turn_PP,
    Count
};

//! Enum of different possible helix positions' types.
enum class HelixPositions : std::size_t
{
    None = 0,
    Start,
    Middle,
    End,
    Start_AND_End,
    Count
};

//! Enum of bridges' types.
enum class BridgeTypes : std::size_t
{
    None = 0,
    AntiParallelBridge,
    ParallelBridge,
    Count
};

/*! \brief
 * Enum of various modes of use of hydrogen atoms. Gromacs mode strictly uses hydrogen atoms from the protein structure,
 * while Dssp mode exclusively uses non-existent hydrogen atoms with coordinates calculated from the positions of carbon and oxygen atoms in residues.
 */
enum class HydrogenMode : std::size_t
{
    Gromacs,
    Dssp,
    Count
};

//! String values corresponding to hydrogen-assignment modes.
const gmx::EnumerationArray<HydrogenMode, const char*> c_HydrogenModeNames = { { "gromacs",
                                                                                 "dssp" } };


/*! \brief
 * Enum of various size of strech of polyproline helices.
 */
enum class PPStretches : std::size_t
{
    Shortened,
    Default,
    Count
};

//! String values corresponding to neighbour-search modes.
const gmx::EnumerationArray<PPStretches, const char*> c_PPStretchesNames = { { "shortened",
                                                                               "default" } };

/*! \brief
 * Describes and manipulates secondary structure attributes of a residue.
 */
class SecondaryStructuresData
{
public:
    /*! \brief
     * Function that sets status of specific secondary structure to a residue.
     */
    void setSecondaryStructureType(SecondaryStructureTypes secondaryStructureTypeName);
    /*! \brief
     * Function that sets status of specific helix position of specific turns' type to a residue.
     */
    void setHelixPosition(HelixPositions helixPosition, TurnsTypes turn);
    /*! \brief
     * Function that sets status "Break" to a residue and it's break partner.
     */
    void setBreak(SecondaryStructuresData* breakPartner);
    /*! \brief
     * Function that sets status "Bridge" or "Anti-Bridge" to a residue and it's bridge partner.
     */
    void setBridge(std::size_t bridgePartnerIndex, BridgeTypes bridgeType);
    /*! \brief
     * Function that returns array of residue's bridges indexes.
     */
    std::vector<std::size_t> const& getBridges(BridgeTypes bridgeType);
    /*! \brief
     * Function that returns boolean status of break existence with another specific residue.
     */
    bool isBreakPartnerWith(const SecondaryStructuresData* partner) const;
    /*! \brief
     * Function that returns boolean status of bridge existence with another specific residue.
     */
    bool hasBridges(BridgeTypes bridgeType) const;
    /*! \brief
     * Function that returns helix position's status of specific turns' type in a residue.
     */
    HelixPositions getHelixPosition(TurnsTypes turn) const;
    /*! \brief
     * Function that returns status of specific secondary structure in a residue.
     */
    SecondaryStructureTypes getSecondaryStructure() const;

private:
    //! Array of pointers to other residues that forms breaks with this residue.
    SecondaryStructuresData* breakPartners_[2] = { nullptr, nullptr };
    //! Array of other residues indexes that forms parralel bridges with this residue.
    std::vector<std::size_t> parallelBridgePartners_;
    //! Array of other residues indexes that forms antiparallel bridges with this residue.
    std::vector<std::size_t> antiBridgePartners_;
    //! Secondary structure's status of this residue.
    SecondaryStructureTypes secondaryStructureLatestStatus_ = SecondaryStructureTypes::Loop;
    //! Array of helix positions from different helix types of this residue.
    std::array<HelixPositions, static_cast<std::size_t>(TurnsTypes::Count)> turnsStatusArray_{
        HelixPositions::None,
        HelixPositions::None,
        HelixPositions::None,
        HelixPositions::None
    };
};

void SecondaryStructuresData::setSecondaryStructureType(const SecondaryStructureTypes secondaryStructureTypeName)
{
    secondaryStructureLatestStatus_ = secondaryStructureTypeName;
}

void SecondaryStructuresData::setHelixPosition(const HelixPositions helixPosition, const TurnsTypes turn)
{
    turnsStatusArray_[static_cast<std::size_t>(turn)] = helixPosition;
}

bool SecondaryStructuresData::isBreakPartnerWith(const SecondaryStructuresData* partner) const
{
    return breakPartners_[0] == partner || breakPartners_[1] == partner;
}

HelixPositions SecondaryStructuresData::getHelixPosition(const TurnsTypes turn) const
{
    return turnsStatusArray_[static_cast<std::size_t>(turn)];
}

SecondaryStructureTypes SecondaryStructuresData::getSecondaryStructure() const
{
    return secondaryStructureLatestStatus_;
}

void SecondaryStructuresData::setBreak(SecondaryStructuresData* breakPartner)
{
    if (breakPartners_[0] == nullptr)
    {
        breakPartners_[0] = breakPartner;
    }
    else
    {
        breakPartners_[1] = breakPartner;
    }
    setSecondaryStructureType(SecondaryStructureTypes::Break);
}

void SecondaryStructuresData::setBridge(std::size_t bridgePartnerIndex, BridgeTypes bridgeType)
{
    if (bridgeType == BridgeTypes::ParallelBridge)
    {
        parallelBridgePartners_.emplace_back(bridgePartnerIndex);
    }
    else
    {
        antiBridgePartners_.emplace_back(bridgePartnerIndex);
    }
}

std::vector<std::size_t> const& SecondaryStructuresData::getBridges(BridgeTypes bridgeType)
{
    if (bridgeType == BridgeTypes::ParallelBridge)
    {
        return parallelBridgePartners_;
    }
    else
    {
        return antiBridgePartners_;
    }
}

bool SecondaryStructuresData::hasBridges(BridgeTypes bridgeType) const
{
    if (bridgeType == BridgeTypes::ParallelBridge)
    {
        return !parallelBridgePartners_.empty();
    }
    else
    {
        return !antiBridgePartners_.empty();
    }
}

/*! \brief
 * Class that provides search of specific h-bond patterns within residues.
 */
class SecondaryStructures
{
public:
    /*! \brief
     * Function that parses topology to construct vector containing information about the residues.
     */
    void analyseTopology(const TopologyInformation& top, Selection const& sel_, HydrogenMode const& transferedHMode);
    /*! \brief
     * Function that checks if ResVector_ is empty. Used after parsing topology data. If it is empty
     * after running analyseTopology(), then some error has occurred.
     */
    bool tolopogyIsIncorrect() const;
    /*! \brief
     * Complex function that provides h-bond patterns search and returns string of one-letter secondary structure definitions.
     */
    std::string performPatternSearch(const t_trxframe& fr,
                                     t_pbc*            pbc,
                                     bool              transferednBSmode_,
                                     real              tranferedCutoff,
                                     bool              transferedPiHelicesPreference,
                                     PPStretches       transferedPolyProStretch);

private:
    //! Vector that contains h-bond pattern information-manipulating class for each residue in selection.
    std::vector<SecondaryStructuresData> secondaryStructuresStatusVector_;
    //! Vector of ResInfo struct that contains all important information about residues in the protein structure.
    std::vector<ResInfo> resVector_;
    //! Function that parses information from a frame to determine hydrogen bonds patterns.
    void analyzeHydrogenBondsInFrame(const t_trxframe& fr, t_pbc* pbc, bool nBSmode_, real cutoff);
    //! Constant float value of h-bond energy. If h-bond energy within residues is smaller than that value, then h-bond exists.
    const float hBondEnergyCutOff_ = -0.5;
    //! Boolean value that indicates the priority of calculating pi-helices.
    bool piHelixPreference_ = false;
    //! String that contains result of dssp calculations for output.
    std::string secondaryStructuresStringLine_;
    /*! \brief
     * Function that provides a simple test if a h-bond exists within two residues of specific indices.
     */
    bool hasHBondBetween(std::size_t Donor, std::size_t Acceptor) const;
    /*! \brief
     * Function that provides a simple test if a chain break exists within two residues of specific indices.
     */
    bool NoChainBreaksBetween(std::size_t residueA, std::size_t residueB) const;
    /*! \brief
     * Function that calculates if bridge or anti-bridge exists within two residues of specific indices.
     */
    BridgeTypes calculateBridge(std::size_t residueA, std::size_t residueB) const;
    /*! \brief
     * Complex function that provides h-bond patterns search of bridges and strands. Part of patternSearch() complex function.
     */
    void analyzeBridgesAndStrandsPatterns();
    /*! \brief
     * Complex function that provides h-bond patterns search of turns and helices. Part of patternSearch() complex function.
     */
    void analyzeTurnsAndHelicesPatterns();
    //! Constant float value that determines the minimum possible distance (in Å) between two Ca atoms of amino acids of the protein,
    //! exceeding which a hydrogen bond between these two residues will be impossible.
    const float minimalCAdistance_ = 9.0;
    //! Enum value for creating hydrogen atoms mode. Very useful for structures without hydrogen atoms. Sets in initial options.
    HydrogenMode hMode_ = HydrogenMode::Gromacs;
    //! Enum value that defines polyproline helix stretch. Can be only equal to 2 or 3. Sets in initial options.
    PPStretches polyProStretch_ = PPStretches::Default;
    /*! \brief
     * Function that calculates atomic distances between atoms A and B based on atom indices.
     */
    static float s_CalculateAtomicDistances(const std::size_t& atomA,
                                            const std::size_t& atomB,
                                            const t_trxframe&  fr,
                                            const t_pbc*       pbc);
    /*! \brief
     * Function that calculates atomic distances between atoms A and B based on atom indices (for atom B) and atom coordinates (for atom A).
     */
    static float s_CalculateAtomicDistances(const rvec&        atomA,
                                            const std::size_t& atomB,
                                            const t_trxframe&  fr,
                                            const t_pbc*       pbc);
    /*! \brief
     * Function that calculates Dihedral Angles based on atom indices.
     */
    static float s_CalculateDihedralAngle(const int&        atomA,
                                          const int&        atomB,
                                          const int&        atomC,
                                          const int&        atomD,
                                          const t_trxframe& fr,
                                          const t_pbc*      pbc);
    /*! \brief
     * Function that calculates dihedral angles in secondary structure map.
     */
    void calculateDihedrals(const t_trxframe& fr, const t_pbc* pbc);
    /*! \brief
     * Function that calculates bends and breaks in secondary structure map.
     */
    void calculateBends(const t_trxframe& fr, const t_pbc* pbc);

    /*! \brief
     * Function that Checks if H-Bond exist according to DSSP algo
     * kCouplingConstant = 27.888,  //  = 332 * 0.42 * 0.2
     * E = k * (1/rON + 1/rCH - 1/rOH - 1/rCN) where CO comes from one AA and NH from another
     * if R is in A
     * Hbond exists if E < -0.5
     */
    void calculateHBondEnergy(ResInfo* Donor, ResInfo* Acceptor, const t_trxframe& fr, const t_pbc* pbc);
};

void SecondaryStructures::analyseTopology(const TopologyInformation& top,
                                          Selection const&           sel_,
                                          HydrogenMode const&        transferedHMode)
{
    hMode_ = transferedHMode;
    int resicompare =
            top.atoms()->atom[static_cast<std::size_t>(*(sel_.atomIndices().begin()))].resind - 1;
    for (gmx::ArrayRef<const int>::iterator ai = sel_.atomIndices().begin();
         ai != sel_.atomIndices().end();
         ++ai)
    {
        if (resicompare != top.atoms()->atom[static_cast<std::size_t>(*ai)].resind)
        {
            resicompare = top.atoms()->atom[static_cast<std::size_t>(*ai)].resind;
            resVector_.emplace_back();
            resVector_.back().info_ = &(top.atoms()->resinfo[resicompare]);
            std::string residueName = *(resVector_.back().info_->name);
            if (residueName == "PRO")
            {
                resVector_.back().isProline_ = true;
            }
        }
        std::string atomName(*(top.atoms()->atomname[static_cast<std::size_t>(*ai)]));
        if (atomName == c_backboneAtomTypeNames[BackboneAtomTypes::AtomCA])
        {
            resVector_.back().backboneIndices_[static_cast<std::size_t>(BackboneAtomTypes::AtomCA)] = *ai;
            resVector_.back().backboneIndicesStatus_[static_cast<std::size_t>(BackboneAtomTypes::AtomCA)] =
                    true;
        }
        else if (atomName == c_backboneAtomTypeNames[BackboneAtomTypes::AtomC])
        {
            resVector_.back().backboneIndices_[static_cast<std::size_t>(BackboneAtomTypes::AtomC)] = *ai;
            resVector_.back().backboneIndicesStatus_[static_cast<std::size_t>(BackboneAtomTypes::AtomC)] =
                    true;
        }
        else if (atomName == c_backboneAtomTypeNames[BackboneAtomTypes::AtomO])
        {
            resVector_.back().backboneIndices_[static_cast<std::size_t>(BackboneAtomTypes::AtomO)] = *ai;
            resVector_.back().backboneIndicesStatus_[static_cast<std::size_t>(BackboneAtomTypes::AtomO)] =
                    true;
        }
        else if (atomName == c_backboneAtomTypeNames[BackboneAtomTypes::AtomN])
        {
            resVector_.back().backboneIndices_[static_cast<std::size_t>(BackboneAtomTypes::AtomN)] = *ai;
            resVector_.back().backboneIndicesStatus_[static_cast<std::size_t>(BackboneAtomTypes::AtomN)] =
                    true;
            if (hMode_ == HydrogenMode::Dssp)
            {
                resVector_.back().backboneIndices_[static_cast<std::size_t>(BackboneAtomTypes::AtomH)] =
                        *ai;
                resVector_.back().backboneIndicesStatus_[static_cast<std::size_t>(BackboneAtomTypes::AtomH)] =
                        true;
            }
        }
        else if (hMode_ == HydrogenMode::Gromacs
                 && atomName == c_backboneAtomTypeNames[BackboneAtomTypes::AtomH])
        {
            resVector_.back().backboneIndices_[static_cast<std::size_t>(BackboneAtomTypes::AtomH)] = *ai;
            resVector_.back().backboneIndicesStatus_[static_cast<std::size_t>(BackboneAtomTypes::AtomH)] =
                    true;
        }
    }
    auto isCorrupted = [](const ResInfo& Res) -> bool {
        return !Res.backboneIndicesStatus_[static_cast<std::size_t>(BackboneAtomTypes::AtomCA)]
               || !Res.backboneIndicesStatus_[static_cast<std::size_t>(BackboneAtomTypes::AtomC)]
               || !Res.backboneIndicesStatus_[static_cast<std::size_t>(BackboneAtomTypes::AtomO)]
               || !Res.backboneIndicesStatus_[static_cast<std::size_t>(BackboneAtomTypes::AtomN)]
               || !Res.backboneIndicesStatus_[static_cast<std::size_t>(BackboneAtomTypes::AtomH)];
    };
    auto corruptedResis = remove_if(resVector_.begin(), resVector_.end(), isCorrupted);
    resVector_.erase(corruptedResis, resVector_.end());
    for (std::size_t i = 1; i < resVector_.size(); ++i)
    {
        resVector_[i].prevResi_     = &(resVector_[i - 1]);
        resVector_[i - 1].nextResi_ = &(resVector_[i]);
    }
}

bool SecondaryStructures::tolopogyIsIncorrect() const
{
    return (resVector_.empty());
}

void SecondaryStructures::analyzeHydrogenBondsInFrame(const t_trxframe& fr, t_pbc* pbc, bool nBSmode_, real cutoff_)
{
    if (nBSmode_)
    {
        std::vector<gmx::RVec> positionsCA_;
        for (std::size_t i = 0; i < resVector_.size(); ++i)
        {
            positionsCA_.emplace_back(fr.x[resVector_[i].getIndex(BackboneAtomTypes::AtomCA)]);
        }
        AnalysisNeighborhood nb_;
        nb_.setCutoff(cutoff_);
        AnalysisNeighborhoodPositions       nbPos_(positionsCA_);
        gmx::AnalysisNeighborhoodSearch     start      = nb_.initSearch(pbc, nbPos_);
        gmx::AnalysisNeighborhoodPairSearch pairSearch = start.startPairSearch(nbPos_);
        gmx::AnalysisNeighborhoodPair       pair;
        ResInfo*                            donor;
        ResInfo*                            acceptor;
        while (pairSearch.findNextPair(&pair))
        {
            if (pair.refIndex() < pair.testIndex())
            {
                donor    = &resVector_[pair.refIndex()];
                acceptor = &resVector_[pair.testIndex()];
            }
            else
            {
                continue;
            }
            calculateHBondEnergy(donor, acceptor, fr, pbc);
            if (acceptor != donor->nextResi_)
            {
                calculateHBondEnergy(acceptor, donor, fr, pbc);
            }
        }
    }
    else
    {
        for (std::size_t donor = 0; donor + 1 < resVector_.size(); ++donor)
        {
            for (std::size_t acceptor = donor + 1; acceptor < resVector_.size(); ++acceptor)
            {
                if (s_CalculateAtomicDistances(resVector_[donor].getIndex(BackboneAtomTypes::AtomCA),
                                               resVector_[acceptor].getIndex(BackboneAtomTypes::AtomCA),
                                               fr,
                                               pbc)
                    < minimalCAdistance_)
                {
                    calculateHBondEnergy(&resVector_[donor], &resVector_[acceptor], fr, pbc);
                    if (acceptor != donor + 1)
                    {
                        calculateHBondEnergy(&resVector_[acceptor], &resVector_[donor], fr, pbc);
                    }
                }
            }
        }
    }
}


bool SecondaryStructures::hasHBondBetween(std::size_t donor, std::size_t acceptor) const
{
    return ((resVector_[donor].acceptor_[0] == resVector_[acceptor].info_
             && resVector_[donor].acceptorEnergy_[0] < hBondEnergyCutOff_)
            || (resVector_[donor].acceptor_[1] == resVector_[acceptor].info_
                && resVector_[donor].acceptorEnergy_[1] < hBondEnergyCutOff_));
}

bool SecondaryStructures::NoChainBreaksBetween(std::size_t residueA, std::size_t residueB) const
{
    if (residueA > residueB)
    {
        std::swap(residueA, residueB);
    }
    for (; residueA != residueB; ++residueA)
    {
        if (secondaryStructuresStatusVector_[residueA].isBreakPartnerWith(
                    &secondaryStructuresStatusVector_[residueA + 1])
            && secondaryStructuresStatusVector_[residueA + 1].isBreakPartnerWith(
                    &secondaryStructuresStatusVector_[residueA]))
        {
            return false;
        }
    }
    return true;
}

BridgeTypes SecondaryStructures::calculateBridge(std::size_t residueA, std::size_t residueB) const
{
    if (residueA < 1 || residueB < 1 || residueA + 1 >= resVector_.size()
        || residueB + 1 >= resVector_.size())
    {
        return BridgeTypes::None;
    }
    if (NoChainBreaksBetween(residueA - 1, residueA + 1)
        && NoChainBreaksBetween(residueB - 1, residueB + 1) && resVector_[residueA].prevResi_
        && resVector_[residueA].nextResi_ && resVector_[residueB].prevResi_ && resVector_[residueB].nextResi_)
    {
        if ((hasHBondBetween(residueA + 1, residueB) && hasHBondBetween(residueB, residueA - 1))
            || (hasHBondBetween(residueB + 1, residueA) && hasHBondBetween(residueA, residueB - 1)))
        {
            return BridgeTypes::ParallelBridge;
        }
        else if ((hasHBondBetween(residueA + 1, residueB - 1) && hasHBondBetween(residueB + 1, residueA - 1))
                 || (hasHBondBetween(residueB, residueA) && hasHBondBetween(residueA, residueB)))
        {
            return BridgeTypes::AntiParallelBridge;
        }
        else
        {
            return BridgeTypes::None;
        }
    }
    return BridgeTypes::None;
}

void SecondaryStructures::analyzeBridgesAndStrandsPatterns()
{
    for (std::size_t i = 1; i + 4 < secondaryStructuresStatusVector_.size(); ++i)
    {
        for (std::size_t j = i + 3; j + 1 < secondaryStructuresStatusVector_.size(); ++j)
        {
            switch (calculateBridge(i, j))
            {
                case BridgeTypes::ParallelBridge:
                {
                    secondaryStructuresStatusVector_[i].setBridge(j, BridgeTypes::ParallelBridge);
                    secondaryStructuresStatusVector_[j].setBridge(i, BridgeTypes::ParallelBridge);
                    break;
                }
                case BridgeTypes::AntiParallelBridge:
                {
                    secondaryStructuresStatusVector_[i].setBridge(j, BridgeTypes::AntiParallelBridge);
                    secondaryStructuresStatusVector_[j].setBridge(i, BridgeTypes::AntiParallelBridge);
                    break;
                }
                default: continue;
            }
        }
    }
    for (std::size_t i = 1; i + 1 < secondaryStructuresStatusVector_.size(); ++i)
    {
        for (std::size_t j = 1; j < 3 and i + j < secondaryStructuresStatusVector_.size(); ++j)
        {
            for (const BridgeTypes& bridgeType :
                 { BridgeTypes::ParallelBridge, BridgeTypes::AntiParallelBridge })
            {
                if (secondaryStructuresStatusVector_[i].hasBridges(bridgeType)
                    && secondaryStructuresStatusVector_[i + j].hasBridges(bridgeType)
                    && (NoChainBreaksBetween(i - 1, i + 1) && NoChainBreaksBetween(i + j - 1, i + j + 1)))
                {
                    std::vector<std::size_t> iPartners =
                            secondaryStructuresStatusVector_[i].getBridges(bridgeType);
                    std::vector<std::size_t> jPartners =
                            secondaryStructuresStatusVector_[i + j].getBridges(bridgeType);
                    for (std::vector<std::size_t>::iterator iPartner = iPartners.begin();
                         iPartner != iPartners.end();
                         ++iPartner)
                    {

                        for (std::vector<std::size_t>::iterator jPartner = jPartners.begin();
                             jPartner != jPartners.end();
                             ++jPartner)
                        {

                            int delta = abs(static_cast<int>(*iPartner) - static_cast<int>(*jPartner));
                            if (delta < 6)
                            {
                                int secondStrandStart = *iPartner;
                                int secondStrandEnd   = *jPartner;
                                if (secondStrandStart > secondStrandEnd)
                                {
                                    std::swap(secondStrandStart, secondStrandEnd);
                                }
                                for (std::size_t k = secondStrandStart;
                                     k <= static_cast<std::size_t>(secondStrandEnd);
                                     ++k)
                                {
                                    secondaryStructuresStatusVector_[k].setSecondaryStructureType(
                                            SecondaryStructureTypes::Strand);
                                }
                                for (std::size_t k = 0; k <= j; ++k)
                                {
                                    secondaryStructuresStatusVector_[i + k].setSecondaryStructureType(
                                            SecondaryStructureTypes::Strand);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (std::size_t i = 1; i + 1 < secondaryStructuresStatusVector_.size(); ++i)
    {
        if (!(secondaryStructuresStatusVector_[i].getSecondaryStructure() == SecondaryStructureTypes::Strand)
            && (secondaryStructuresStatusVector_[i].hasBridges(BridgeTypes::ParallelBridge)
                || secondaryStructuresStatusVector_[i].hasBridges(BridgeTypes::AntiParallelBridge)))
        {
            secondaryStructuresStatusVector_[i].setSecondaryStructureType(SecondaryStructureTypes::Bridge);
        }
    }
}

void SecondaryStructures::analyzeTurnsAndHelicesPatterns()
{
    for (const TurnsTypes& i : { TurnsTypes::Turn_3, TurnsTypes::Turn_4, TurnsTypes::Turn_5 })
    {
        std::size_t stride = static_cast<std::size_t>(i) + 3;
        for (std::size_t j = 0; j + stride < secondaryStructuresStatusVector_.size(); ++j)
        {
            if (hasHBondBetween(j + stride, j) && NoChainBreaksBetween(j, j + stride))
            {
                secondaryStructuresStatusVector_[j + stride].setHelixPosition(HelixPositions::End, i);

                for (std::size_t k = 1; k < stride; ++k)
                {
                    if (secondaryStructuresStatusVector_[j + k].getHelixPosition(i) == HelixPositions::None)
                    {
                        secondaryStructuresStatusVector_[j + k].setHelixPosition(HelixPositions::Middle, i);
                    }
                }
                if (secondaryStructuresStatusVector_[j].getHelixPosition(i) == HelixPositions::End)
                {
                    secondaryStructuresStatusVector_[j].setHelixPosition(HelixPositions::Start_AND_End, i);
                }
                else
                {
                    secondaryStructuresStatusVector_[j].setHelixPosition(HelixPositions::Start, i);
                }
            }
        }
    }

    for (const TurnsTypes& i : { TurnsTypes::Turn_4, TurnsTypes::Turn_3, TurnsTypes::Turn_5 })
    {
        std::size_t stride = static_cast<std::size_t>(i) + 3;
        for (std::size_t j = 1; j + stride < secondaryStructuresStatusVector_.size(); ++j)
        {
            if ((secondaryStructuresStatusVector_[j - 1].getHelixPosition(i) == HelixPositions::Start
                 || secondaryStructuresStatusVector_[j - 1].getHelixPosition(i) == HelixPositions::Start_AND_End)
                && (secondaryStructuresStatusVector_[j].getHelixPosition(i) == HelixPositions::Start
                    || secondaryStructuresStatusVector_[j].getHelixPosition(i) == HelixPositions::Start_AND_End))
            {
                bool                    empty = true;
                SecondaryStructureTypes helix;
                switch (i)
                {
                    case TurnsTypes::Turn_3:
                        for (std::size_t k = 0; empty && k < stride; ++k)
                        {
                            empty = secondaryStructuresStatusVector_[j + k].getSecondaryStructure()
                                    <= SecondaryStructureTypes::Helix_3;
                        }
                        helix = SecondaryStructureTypes::Helix_3;
                        break;
                    case TurnsTypes::Turn_5:
                        for (std::size_t k = 0; empty && k < stride; ++k)
                        {
                            empty = secondaryStructuresStatusVector_[j + k].getSecondaryStructure()
                                            <= SecondaryStructureTypes::Helix_5
                                    || (piHelixPreference_
                                        && secondaryStructuresStatusVector_[j + k].getSecondaryStructure()
                                                   == SecondaryStructureTypes::Helix_4);
                        }
                        helix = SecondaryStructureTypes::Helix_5;
                        break;
                    default: helix = SecondaryStructureTypes::Helix_4; break;
                }
                if (empty || helix == SecondaryStructureTypes::Helix_4)
                {
                    for (std::size_t k = 0; k < stride; ++k)
                    {
                        secondaryStructuresStatusVector_[j + k].setSecondaryStructureType(helix);
                    }
                }
            }
        }
    }
    for (std::size_t i = 1; i + 1 < secondaryStructuresStatusVector_.size(); ++i)
    {
        if (secondaryStructuresStatusVector_[i].getSecondaryStructure() <= SecondaryStructureTypes::Turn)
        {
            bool isTurn = false;
            for (const TurnsTypes& j : { TurnsTypes::Turn_3, TurnsTypes::Turn_4, TurnsTypes::Turn_5 })
            {
                std::size_t stride = static_cast<std::size_t>(j) + 3;
                for (std::size_t k = 1; k < stride and !isTurn; ++k)
                {
                    isTurn = (i >= k)
                             && (secondaryStructuresStatusVector_[i - k].getHelixPosition(j)
                                         == HelixPositions::Start
                                 || secondaryStructuresStatusVector_[i - k].getHelixPosition(j)
                                            == HelixPositions::Start_AND_End);
                }
            }
            if (isTurn)
            {
                secondaryStructuresStatusVector_[i].setSecondaryStructureType(SecondaryStructureTypes::Turn);
            }
        }
    }
}

std::string SecondaryStructures::performPatternSearch(const t_trxframe& fr,
                                                      t_pbc*            pbc,
                                                      bool              transferednBSmode_,
                                                      real              tranferedCutoff,
                                                      bool        transferedPiHelicesPreference,
                                                      PPStretches transferedPolyProStretch)
{
    if (resVector_.empty())
    {
        gmx_fatal(
                FARGS,
                "Invalid usage of this function. You have to load topology information before. Run "
                "analyseTopology(...) first.");
    }
    analyzeHydrogenBondsInFrame(fr, pbc, transferednBSmode_, tranferedCutoff);
    piHelixPreference_ = transferedPiHelicesPreference;
    polyProStretch_    = transferedPolyProStretch;
    secondaryStructuresStatusVector_.resize(resVector_.size());
    secondaryStructuresStringLine_.resize(resVector_.size(), '~');
    calculateBends(fr, pbc);
    calculateDihedrals(fr, pbc);
    analyzeBridgesAndStrandsPatterns();
    analyzeTurnsAndHelicesPatterns();
    for (auto i = static_cast<std::size_t>(SecondaryStructureTypes::Bend);
         i != static_cast<std::size_t>(SecondaryStructureTypes::Count);
         ++i)
    {
        for (std::size_t j = 0; j < secondaryStructuresStatusVector_.size(); ++j)
        {
            if (secondaryStructuresStatusVector_[j].getSecondaryStructure()
                == static_cast<SecondaryStructureTypes>(i))
            {
                secondaryStructuresStringLine_[j] = c_secondaryStructureTypeNames[i];
            }
        }
    }
    if (piHelixPreference_)
    {
        for (std::size_t j = 0; j < secondaryStructuresStatusVector_.size(); ++j)
        {
            if (secondaryStructuresStatusVector_[j].getSecondaryStructure() == SecondaryStructureTypes::Helix_5
                && secondaryStructuresStatusVector_[j].getSecondaryStructure()
                           == SecondaryStructureTypes::Helix_4)
            {
                secondaryStructuresStringLine_[j] =
                        c_secondaryStructureTypeNames[SecondaryStructureTypes::Helix_5];
            }
        }
    }
    if (secondaryStructuresStatusVector_.size() > 1)
    {
        for (std::size_t i = 0, lineFactor = 1; i + 1 < secondaryStructuresStatusVector_.size(); ++i)
        {
            if (secondaryStructuresStatusVector_[i].getSecondaryStructure() == SecondaryStructureTypes::Break
                && secondaryStructuresStatusVector_[i + 1].getSecondaryStructure()
                           == SecondaryStructureTypes::Break)
            {
                if (secondaryStructuresStatusVector_[i].isBreakPartnerWith(
                            &secondaryStructuresStatusVector_[i + 1])
                    && secondaryStructuresStatusVector_[i + 1].isBreakPartnerWith(
                            &secondaryStructuresStatusVector_[i]))
                {
                    secondaryStructuresStringLine_.insert(
                            secondaryStructuresStringLine_.begin() + i + lineFactor,
                            c_secondaryStructureTypeNames[SecondaryStructureTypes::Break]);
                    ++lineFactor;
                }
            }
        }
    }
    return secondaryStructuresStringLine_;
}

float SecondaryStructures::s_CalculateAtomicDistances(const std::size_t& atomA,
                                                      const std::size_t& atomB,
                                                      const t_trxframe&  fr,
                                                      const t_pbc*       pbc)
{
    gmx::RVec vectorBA = { 0, 0, 0 };
    pbc_dx(pbc, fr.x[atomA], fr.x[atomB], vectorBA.as_vec());
    return vectorBA.norm() * gmx::c_nm2A;
}

float SecondaryStructures::s_CalculateAtomicDistances(const rvec&        atomA,
                                                      const std::size_t& atomB,
                                                      const t_trxframe&  fr,
                                                      const t_pbc*       pbc)
{
    gmx::RVec vectorBA = { 0, 0, 0 };
    pbc_dx(pbc, atomA, fr.x[atomB], vectorBA.as_vec());
    return vectorBA.norm() * gmx::c_nm2A;
}

float SecondaryStructures::s_CalculateDihedralAngle(const int&        atomA,
                                                    const int&        atomB,
                                                    const int&        atomC,
                                                    const int&        atomD,
                                                    const t_trxframe& fr,
                                                    const t_pbc*      pbc)
{
    float     result               = 360;
    float     vdist1               = 0;
    float     vdist2               = 0;
    gmx::RVec vectorBA             = { 0, 0, 0 };
    gmx::RVec vectorCD             = { 0, 0, 0 };
    gmx::RVec vectorCB             = { 0, 0, 0 };
    gmx::RVec vectorCBxBA          = { 0, 0, 0 };
    gmx::RVec vectorCBxCD          = { 0, 0, 0 };
    gmx::RVec vectorCBxvectorCBxCD = { 0, 0, 0 };
    pbc_dx(pbc, fr.x[atomA], fr.x[atomB], vectorBA.as_vec());
    pbc_dx(pbc, fr.x[atomD], fr.x[atomC], vectorCD.as_vec());
    pbc_dx(pbc, fr.x[atomB], fr.x[atomC], vectorCB.as_vec());
    vectorBA *= gmx::c_nm2A;
    vectorCD *= gmx::c_nm2A;
    vectorCB *= gmx::c_nm2A;
    for (std::size_t i = XX, j = i + 1, k = i + 2; i <= ZZ; ++i, ++j, ++k)
    {
        if (j > 2)
        {
            j -= 3;
        }
        if (k > 2)
        {
            k -= 3;
        }
        vectorCBxBA[i] = (vectorCB[j] * vectorBA[k]) - (vectorCB[k] * vectorBA[j]);
        vectorCBxCD[i] = (vectorCB[j] * vectorCD[k]) - (vectorCB[k] * vectorCD[j]);
    }
    for (std::size_t i = XX, j = i + 1, k = i + 2; i <= ZZ; ++i, ++j, ++k)
    {
        if (j > 2)
        {
            j -= 3;
        }
        if (k > 2)
        {
            k -= 3;
        }
        vectorCBxvectorCBxCD[i] = (vectorCB[j] * vectorCBxCD[k]) - (vectorCB[k] * vectorCBxCD[j]);
    }
    vdist1 = (vectorCBxCD[XX] * vectorCBxCD[XX]) + (vectorCBxCD[YY] * vectorCBxCD[YY])
             + (vectorCBxCD[ZZ] * vectorCBxCD[ZZ]);
    vdist2 = (vectorCBxvectorCBxCD[XX] * vectorCBxvectorCBxCD[XX])
             + (vectorCBxvectorCBxCD[YY] * vectorCBxvectorCBxCD[YY])
             + (vectorCBxvectorCBxCD[ZZ] * vectorCBxvectorCBxCD[ZZ]);
    if (vdist1 > 0 and vdist2 > 0)
    {
        vdist1 = ((vectorCBxBA[XX] * vectorCBxCD[XX]) + (vectorCBxBA[YY] * vectorCBxCD[YY])
                  + (vectorCBxBA[ZZ] * vectorCBxCD[ZZ]))
                 / std::sqrt(vdist1);
        vdist2 = ((vectorCBxBA[XX] * vectorCBxvectorCBxCD[XX])
                  + (vectorCBxBA[YY] * vectorCBxvectorCBxCD[YY])
                  + (vectorCBxBA[ZZ] * vectorCBxvectorCBxCD[ZZ]))
                 / std::sqrt(vdist2);
        if (vdist1 != 0 or vdist2 != 0)
        {
            result = std::atan2(vdist2, vdist1) * gmx::c_rad2Deg;
        }
    }
    return result;
}

void SecondaryStructures::calculateDihedrals(const t_trxframe& fr, const t_pbc* pbc)
{
    const float        epsilon = 29;
    const float        phiMin  = -75 - epsilon;
    const float        phiMax  = -75 + epsilon;
    const float        psiMin  = 145 - epsilon;
    const float        psiMax  = 145 + epsilon;
    std::vector<float> phi(resVector_.size(), 360);
    std::vector<float> psi(resVector_.size(), 360);
    for (std::size_t i = 1; i + 1 < resVector_.size(); ++i)
    {
        phi[i] = s_CalculateDihedralAngle(resVector_[i - 1].getIndex(BackboneAtomTypes::AtomC),
                                          resVector_[i].getIndex(BackboneAtomTypes::AtomN),
                                          resVector_[i].getIndex(BackboneAtomTypes::AtomCA),
                                          resVector_[i].getIndex(BackboneAtomTypes::AtomC),
                                          fr,
                                          pbc);
        psi[i] = s_CalculateDihedralAngle(resVector_[i].getIndex(BackboneAtomTypes::AtomN),
                                          resVector_[i].getIndex(BackboneAtomTypes::AtomCA),
                                          resVector_[i].getIndex(BackboneAtomTypes::AtomC),
                                          resVector_[i + 1].getIndex(BackboneAtomTypes::AtomN),
                                          fr,
                                          pbc);
    }
    for (std::size_t i = 1; i + 3 < resVector_.size(); ++i)
    {
        switch (polyProStretch_)
        {
            case PPStretches::Shortened:
            {
                if (phiMin > phi[i] or phi[i] > phiMax or phiMin > phi[i + 1] or phi[i + 1] > phiMax)
                {
                    continue;
                }

                if (psiMin > psi[i] or psi[i] > psiMax or psiMin > psi[i + 1] or psi[i + 1] > psiMax)
                {
                    continue;
                }

                switch (secondaryStructuresStatusVector_[i].getHelixPosition(TurnsTypes::Turn_PP))
                {
                    case HelixPositions::None:
                        secondaryStructuresStatusVector_[i].setHelixPosition(HelixPositions::Start,
                                                                             TurnsTypes::Turn_PP);
                        break;

                    case HelixPositions::End:
                        secondaryStructuresStatusVector_[i].setHelixPosition(
                                HelixPositions::Start_AND_End, TurnsTypes::Turn_PP);
                        break;

                    default: break;
                }
                secondaryStructuresStatusVector_[i + 1].setHelixPosition(HelixPositions::End,
                                                                         TurnsTypes::Turn_PP);
                if (secondaryStructuresStatusVector_[i].getSecondaryStructure()
                    == SecondaryStructureTypes::Loop)
                {
                    secondaryStructuresStatusVector_[i].setSecondaryStructureType(
                            SecondaryStructureTypes::Helix_PP);
                }
                if (secondaryStructuresStatusVector_[i + 1].getSecondaryStructure()
                    == SecondaryStructureTypes::Loop)
                {
                    secondaryStructuresStatusVector_[i + 1].setSecondaryStructureType(
                            SecondaryStructureTypes::Helix_PP);
                }
                break;
            }
            case PPStretches::Default:
            {
                if (phiMin > phi[i] or phi[i] > phiMax or phiMin > phi[i + 1] or phi[i + 1] > phiMax
                    or phiMin > phi[i + 2] or phi[i + 2] > phiMax)
                {
                    continue;
                }

                if (psiMin > psi[i] or psi[i] > psiMax or psiMin > psi[i + 1] or psi[i + 1] > psiMax
                    or psiMin > psi[i + 2] or psi[i + 2] > psiMax)
                {
                    continue;
                }
                switch (secondaryStructuresStatusVector_[i].getHelixPosition(TurnsTypes::Turn_PP))
                {
                    case HelixPositions::None:
                        secondaryStructuresStatusVector_[i].setHelixPosition(HelixPositions::Start,
                                                                             TurnsTypes::Turn_PP);
                        break;

                    case HelixPositions::End:
                        secondaryStructuresStatusVector_[i].setHelixPosition(
                                HelixPositions::Start_AND_End, TurnsTypes::Turn_PP);
                        break;

                    default: break;
                }
                secondaryStructuresStatusVector_[i + 1].setHelixPosition(HelixPositions::Middle,
                                                                         TurnsTypes::Turn_PP);
                secondaryStructuresStatusVector_[i + 2].setHelixPosition(HelixPositions::End,
                                                                         TurnsTypes::Turn_PP);
                if (secondaryStructuresStatusVector_[i].getSecondaryStructure()
                    == SecondaryStructureTypes::Loop)
                {
                    secondaryStructuresStatusVector_[i].setSecondaryStructureType(
                            SecondaryStructureTypes::Helix_PP);
                }
                if (secondaryStructuresStatusVector_[i + 1].getSecondaryStructure()
                    == SecondaryStructureTypes::Loop)
                {
                    secondaryStructuresStatusVector_[i + 1].setSecondaryStructureType(
                            SecondaryStructureTypes::Helix_PP);
                }
                if (secondaryStructuresStatusVector_[i + 2].getSecondaryStructure()
                    == SecondaryStructureTypes::Loop)
                {
                    secondaryStructuresStatusVector_[i + 2].setSecondaryStructureType(
                            SecondaryStructureTypes::Helix_PP);
                }
                break;
            }
            default: gmx_fatal(FARGS, "Unsupported stretch length");
        }
    }
}

void SecondaryStructures::calculateBends(const t_trxframe& fr, const t_pbc* pbc)
{
    const float bendDegree = 70.0;
    const float maxDist    = 2.5;
    float       degree     = 0;
    float       vdist      = 0;
    float       vprod      = 0;
    gmx::RVec   caPos1{ 0, 0, 0 };
    gmx::RVec   caPos2{ 0, 0, 0 };
    for (std::size_t i = 0; i + 1 < resVector_.size(); ++i)
    {
        if (s_CalculateAtomicDistances(
                    static_cast<int>(resVector_[i].getIndex(BackboneAtomTypes::AtomC)),
                    static_cast<int>(resVector_[i + 1].getIndex(BackboneAtomTypes::AtomN)),
                    fr,
                    pbc)
            > maxDist)
        {
            secondaryStructuresStatusVector_[i].setBreak(&secondaryStructuresStatusVector_[i + 1]);
            secondaryStructuresStatusVector_[i + 1].setBreak(&secondaryStructuresStatusVector_[i]);
        }
    }
    for (std::size_t i = 2; i + 2 < resVector_.size(); ++i)
    {
        if (secondaryStructuresStatusVector_[i - 2].isBreakPartnerWith(
                    &(secondaryStructuresStatusVector_[i - 1]))
            || secondaryStructuresStatusVector_[i - 1].isBreakPartnerWith(
                    &(secondaryStructuresStatusVector_[i]))
            || secondaryStructuresStatusVector_[i].isBreakPartnerWith(
                    &(secondaryStructuresStatusVector_[i + 1]))
            || secondaryStructuresStatusVector_[i + 1].isBreakPartnerWith(
                    &(secondaryStructuresStatusVector_[i + 2])))
        {
            continue;
        }
        for (int j = 0; j < 3; ++j)
        {
            caPos1[j] = fr.x[resVector_[i].getIndex(BackboneAtomTypes::AtomCA)][j]
                        - fr.x[resVector_[i - 2].getIndex(BackboneAtomTypes::AtomCA)][j];
            caPos2[j] = fr.x[resVector_[i + 2].getIndex(BackboneAtomTypes::AtomCA)][j]
                        - fr.x[resVector_[i].getIndex(BackboneAtomTypes::AtomCA)][j];
        }
        vdist = (caPos1[0] * caPos2[0]) + (caPos1[1] * caPos2[1]) + (caPos1[2] * caPos2[2]);
        vprod = s_CalculateAtomicDistances(resVector_[i - 2].getIndex(BackboneAtomTypes::AtomCA),
                                           resVector_[i].getIndex(BackboneAtomTypes::AtomCA),
                                           fr,
                                           pbc)
                * gmx::c_angstrom / gmx::c_nano
                * s_CalculateAtomicDistances(resVector_[i].getIndex(BackboneAtomTypes::AtomCA),
                                             resVector_[i + 2].getIndex(BackboneAtomTypes::AtomCA),
                                             fr,
                                             pbc)
                * gmx::c_angstrom / gmx::c_nano;
        degree = std::acos(vdist / vprod) * gmx::c_rad2Deg;
        if (degree > bendDegree)
        {
            secondaryStructuresStatusVector_[i].setSecondaryStructureType(SecondaryStructureTypes::Bend);
        }
    }
}

void SecondaryStructures::calculateHBondEnergy(ResInfo*          Donor,
                                               ResInfo*          Acceptor,
                                               const t_trxframe& fr,
                                               const t_pbc*      pbc)
{
    const float kCouplingConstant   = 27.888;
    const float minimalAtomDistance = 0.5;
    const float minEnergy           = -9.9;
    float       HbondEnergy         = 0;
    float       distanceNO          = 0;
    float       distanceHC          = 0;
    float       distanceHO          = 0;
    float       distanceNC          = 0;
    if (!(Donor->isProline_)
        && (Acceptor->getIndex(BackboneAtomTypes::AtomC)
            && Acceptor->getIndex(BackboneAtomTypes::AtomO) && Donor->getIndex(BackboneAtomTypes::AtomN)
            && (Donor->getIndex(BackboneAtomTypes::AtomH) || hMode_ == HydrogenMode::Dssp)))
    {
        distanceNO = s_CalculateAtomicDistances(Donor->getIndex(BackboneAtomTypes::AtomN),
                                                Acceptor->getIndex(BackboneAtomTypes::AtomO),
                                                fr,
                                                pbc);
        distanceNC = s_CalculateAtomicDistances(Donor->getIndex(BackboneAtomTypes::AtomN),
                                                Acceptor->getIndex(BackboneAtomTypes::AtomC),
                                                fr,
                                                pbc);
        if (hMode_ == HydrogenMode::Dssp)
        {
            if (Donor->prevResi_ != nullptr && Donor->prevResi_->getIndex(BackboneAtomTypes::AtomC)
                && Donor->prevResi_->getIndex(BackboneAtomTypes::AtomO))
            {
                gmx::RVec atomH  = fr.x[Donor->getIndex(BackboneAtomTypes::AtomH)];
                gmx::RVec prevCO = fr.x[Donor->prevResi_->getIndex(BackboneAtomTypes::AtomC)];
                prevCO -= fr.x[Donor->prevResi_->getIndex(BackboneAtomTypes::AtomO)];
                float prevCODist = s_CalculateAtomicDistances(
                        Donor->prevResi_->getIndex(BackboneAtomTypes::AtomC),
                        Donor->prevResi_->getIndex(BackboneAtomTypes::AtomO),
                        fr,
                        pbc);
                atomH += prevCO / prevCODist;
                distanceHO = s_CalculateAtomicDistances(
                        atomH, Acceptor->getIndex(BackboneAtomTypes::AtomO), fr, pbc);
                distanceHC = s_CalculateAtomicDistances(
                        atomH, Acceptor->getIndex(BackboneAtomTypes::AtomC), fr, pbc);
            }
            else
            {
                distanceHO = distanceNO;
                distanceHC = distanceNC;
            }
        }
        else
        {
            distanceHO = s_CalculateAtomicDistances(Donor->getIndex(BackboneAtomTypes::AtomH),
                                                    Acceptor->getIndex(BackboneAtomTypes::AtomO),
                                                    fr,
                                                    pbc);
            distanceHC = s_CalculateAtomicDistances(Donor->getIndex(BackboneAtomTypes::AtomH),
                                                    Acceptor->getIndex(BackboneAtomTypes::AtomC),
                                                    fr,
                                                    pbc);
        }
        if ((distanceNO < minimalAtomDistance) || (distanceHC < minimalAtomDistance)
            || (distanceHO < minimalAtomDistance) || (distanceNC < minimalAtomDistance))
        {
            HbondEnergy = minEnergy;
        }
        else
        {
            HbondEnergy = kCouplingConstant
                          * ((1 / distanceNO) + (1 / distanceHC) - (1 / distanceHO) - (1 / distanceNC));
        }
    }
    if (HbondEnergy < Donor->acceptorEnergy_[0])
    {
        Donor->acceptor_[1]       = Donor->acceptor_[0];
        Donor->acceptorEnergy_[1] = Donor->acceptorEnergy_[0];
        Donor->acceptor_[0]       = Acceptor->info_;
        Donor->acceptorEnergy_[0] = HbondEnergy;
    }
    else if (HbondEnergy < Donor->acceptorEnergy_[1])
    {
        Donor->acceptor_[1]       = Acceptor->info_;
        Donor->acceptorEnergy_[1] = HbondEnergy;
    }

    if (HbondEnergy < Acceptor->donorEnergy_[0])
    {
        Acceptor->donor_[1]       = Acceptor->donor_[0];
        Acceptor->donorEnergy_[1] = Acceptor->donorEnergy_[0];
        Acceptor->donor_[0]       = Donor->info_;
        Acceptor->donorEnergy_[0] = HbondEnergy;
    }
    else if (HbondEnergy < Acceptor->donorEnergy_[1])
    {
        Acceptor->donor_[1]       = Donor->info_;
        Acceptor->donorEnergy_[1] = HbondEnergy;
    }
}


class Dssp : public TrajectoryAnalysisModule
{
public:
    void initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings) override;
    void optionsFinished(TrajectoryAnalysisSettings* settings) override;
    void initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top) override;
    void initAfterFirstFrame(const TrajectoryAnalysisSettings& settings, const t_trxframe& fr) override;
    void analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata) override;
    void finishAnalysis(int nframes) override;
    void writeOutput() override;

private:
    //! Selections for DSSP output. Sets in initial options.
    Selection sel_;
    //! Boolean value for Preferring P-Helices mode. Sets in initial options.
    bool polyProHelices_ = true;
    //! Enum value for creating hydrogen atoms mode. Very useful for structures without hydrogen atoms. Sets in initial options.
    HydrogenMode hMode_ = HydrogenMode::Gromacs;
    //! Boolean value determines differend calculation methods for searching neighbour residues. Sets in initial options.
    bool nBSmode_ = true;
    //! Real value that defines maximum distance from residue to its neighbour residue.
    real cutoff_ = 0.9;
    //! Enum value that defines polyproline helix stretch. Can be only equal to 2 or 3. Sets in initial options.
    PPStretches polyProStretch_ = PPStretches::Default;
    //! String value that defines output filename. Sets in initial options.
    std::string fnmDSSPOut_ = "dssp.dat";
    //! Class that calculates h-bond patterns in secondary structure map based on original DSSP algo.
    SecondaryStructures patternSearch_;
    //! A storage that contains DSSP info_ from different frames.
    DsspStorage storage_;
};

void Dssp::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{
    static const char* const desc[] = {
        "[THISMODULE] allows using the DSSP algorithm (namely, by detecting specific patterns of "
        "hydrogen bonds between amino acid residues) "
        "to determine the secondary structure of a protein.[PAR]"
        "[TT]-hmode[tt] selects between using hydrogen atoms directly from the structure "
        "(\"gromacs\" option) and using hydrogen pseudo-atoms based on C and O atom coordinates of "
        "previous residue (\"dssp\" option).[PAR]"
        "[TT]-nb[tt] allowns to use gromacs neighbors search method to find residue pairs that may "
        "have a "
        "hydrogen bond instead of simply iterating over the residues among themselves.[PAR]"
        "[TT]-cutoff[tt] is a real value that defines maximum distance from residue to its "
        "neighbour residue."
        "Only makes sense when using with [TT]-nb nbsearch[tt] option. Minimum (and also "
        "recommended)"
        " value is 0,9.[PAR]"
        "[TT]-pihelix[tt] changes pattern-search algorithm towards preference of pi-helices.[PAR]"
        "[TT]-ppstretch[tt] defines strech value of polyproline-helices. \"Shortened\" means "
        "strech with"
        "size 2 and \"Default\" means strech with size 3.[PAR]"
        "Note that [THISMODULE] currently is not capable of reproducing "
        "the secondary structure of proteins whose structure is determined by methods other than "
        "X-ray crystallography (structures in .pdb format with "
        "incorrect values in the CRYST1 line) due to the incorrect cell size in such structures.",
    };
    options->addOption(FileNameOption("o")
                               .outputFile()
                               .store(&fnmDSSPOut_)
                               .required()
                               .defaultBasename("dssp")
                               .filetype(OptionFileType::GenericData)
                               .description("Filename for DSSP output"));
    options->addOption(SelectionOption("sel").store(&sel_).defaultSelectionText("Protein").description(
            "Group for DSSP"));
    options->addOption(EnumOption<HydrogenMode>("hmode")
                               .store(&hMode_)
                               .defaultValue(HydrogenMode::Gromacs)
                               .enumValue(c_HydrogenModeNames)
                               .description("Hydrogens pseudoatoms creating mode"));
    options->addOption(BooleanOption("nb").store(&nBSmode_).defaultValue(true).description(
            "Use gromacs neighbors search method"));
    options->addOption(RealOption("cutoff").store(&cutoff_).required().defaultValue(0.9).description(
            "Distance from residue to its neighbour residue in neighbour search. Must be >= 0,9."));
    options->addOption(
            BooleanOption("pihelix").store(&polyProHelices_).defaultValue(true).description("Prefer Pi Helices"));
    options->addOption(EnumOption<PPStretches>("ppstretch")
                               .store(&polyProStretch_)
                               .defaultValue(PPStretches::Default)
                               .enumValue(c_PPStretchesNames)
                               .description("Stretch value for PP-helices"));
    settings->setHelpText(desc);
}

void Dssp::optionsFinished(TrajectoryAnalysisSettings* /* settings */)
{
    if (cutoff_ < real(0.9))
    {
        gmx_fatal(FARGS, "Invalid cutoff value. It must be >= 0,9.");
    }
}

void Dssp::initAnalysis(const TrajectoryAnalysisSettings& /* settings */, const TopologyInformation& top)
{
    patternSearch_.analyseTopology(top, sel_, hMode_);
}

void Dssp::initAfterFirstFrame(const TrajectoryAnalysisSettings& /* settings */, const t_trxframe& /* fr */)
{
    if (patternSearch_.tolopogyIsIncorrect())
    {
        std::string errorDesc =
                "From these inputs, it is not possible to obtain proper information about the "
                "patterns of hydrogen bonds.";
        if (hMode_ != HydrogenMode::Dssp)
        {
            errorDesc += " Maybe you should add the \"-hmode dssp\" option?";
        }
        gmx_fatal(FARGS, "%s", errorDesc.c_str());
    }
}

void Dssp::analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* /* pdata */)
{
    storage_.addData(frnr,
                     patternSearch_.performPatternSearch(
                             fr, pbc, nBSmode_, cutoff_, polyProHelices_, polyProStretch_));
}

void Dssp::finishAnalysis(int /*nframes*/) {}

void Dssp::writeOutput()
{
    std::vector<DsspStorageFrame> dataOut;
    FILE*                         fp;
    fp      = gmx_ffopen(fnmDSSPOut_, "w");
    dataOut = storage_.getData();
    for (auto& i : dataOut)
    {
        std::fprintf(fp, "%s\n", i.dsspData_.c_str());
    }
    gmx_ffclose(fp);
}

} // namespace

const char DsspInfo::name[]             = "dssp";
const char DsspInfo::shortDescription[] = "Calculate protein secondary structure via DSSP algo";

TrajectoryAnalysisModulePointer DsspInfo::create()
{
    return TrajectoryAnalysisModulePointer(new Dssp);
}

} // namespace analysismodules

} // namespace gmx
