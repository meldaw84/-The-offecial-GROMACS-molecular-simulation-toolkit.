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
    int frnr = 0;
    //! Frame dssp data.
    std::string dsspData;
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
    std::vector<DsspStorageFrame> getData();

private:
    /*! \brief
     * Vector that contains information from different frames.
     */
    std::vector<DsspStorageFrame> data_;
};

void DsspStorage::addData(int frnr, const std::string& data)
{
    DsspStorageFrame dsspData;
    dsspData.frnr     = frnr;
    dsspData.dsspData = data;
    data_.push_back(dsspData);
}

std::vector<DsspStorageFrame> DsspStorage::getData()
{
    return data_;
}

//! Enum of backbone atoms' types.
enum class backboneAtomTypes : std::size_t
{
    AtomCA,
    AtomC,
    AtomO,
    AtomN,
    AtomH,
    Count
};

//! String values corresponding to backbone atom types.
const gmx::EnumerationArray<backboneAtomTypes, const char*> backboneAtomTypeNames = {
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
    std::array<std::size_t, static_cast<std::size_t>(backboneAtomTypes::Count)> _backboneIndices = { 0, 0, 0, 0, 0 };
    /*! \brief
     * Bool array determining whether backbone atoms have been assigned.
     */
    std::array<bool, static_cast<std::size_t>(backboneAtomTypes::Count)>
            _backboneIndicesStatus = { false, false, false, false, false };
    /*! \brief
     * Function that returns atom's index based on specific atom type.
     */
    std::size_t getIndex(backboneAtomTypes atomTypeName) const;
    //! Pointer to t_resinfo which contains full information about this specific residue.
    t_resinfo* info = nullptr;
    //! Pointer to t_resinfo which contains full information about this residue's h-bond donors.
    t_resinfo* donor[2] = { nullptr, nullptr };
    //! Pointer to t_resinfo which contains full information about this residue's h-bond acceptors.
    t_resinfo* acceptor[2] = { nullptr, nullptr };
    //! Pointer to previous residue in list.
    ResInfo* prevResi = nullptr;
    //! Pointer to next residue in list.
    ResInfo* nextResi = nullptr;
    //! Float value of h-bond energy with this residue's donors.
    float donorEnergy[2] = { 0, 0 };
    //! Float value of h-bond energy with this residue's accpetors.
    float acceptorEnergy[2] = { 0, 0 };
    //! Bool value that defines either this residue is proline (PRO) or not.
    bool is_proline = false;
};

std::size_t ResInfo::getIndex(backboneAtomTypes atomTypeName) const
{
    return _backboneIndices[static_cast<std::size_t>(atomTypeName)];
}

//! Enum of secondary structures' types.
enum class secondaryStructureTypes : std::size_t
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
const gmx::EnumerationArray<secondaryStructureTypes, const char> secondaryStructureTypeNames = {
    { '~', '=', 'S', 'T', 'P', 'I', 'G', 'E', 'B', 'H' }
};

//! Enum of turns' types.
enum class turnsTypes : std::size_t
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
enum class bridgeTypes : std::size_t
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
const gmx::EnumerationArray<HydrogenMode, const char*> HydrogenModeNames = { { "gromacs", "dssp" } };

/*! \brief
 * Enum of various modes of finding neighbour residues in structure.
 */
enum class NBSearchMethod : std::size_t
{
    NBSearch,
    Direct,
    Count
};

//! String values corresponding to neighbour-search modes.
const gmx::EnumerationArray<NBSearchMethod, const char*> NBSearchMethodNames = { { "nbsearch",
                                                                                   "direct" } };

/*! \brief
 * Enum of various size of strech of polyproline helices.
 */
enum class PPStreches : std::size_t
{
    Shortened,
    Default,
    Count
};

//! String values corresponding to neighbour-search modes.
const gmx::EnumerationArray<PPStreches, const char*> PPStrechesNames = { { "shortened",
                                                                           "default" } };

/*! \brief
 * Class that precisely manipulates secondary structures' statuses and assigns additional
 * information to the residues required when calculating hydrogen-bond patterns.
 */
class secondaryStructuresData
{
public:
    /*! \brief
     * Function that sets status of specific secondary structure to a residue.
     */
    void setStatus(secondaryStructureTypes secondaryStructureTypeName, bool status = true);
    /*! \brief
     * Function that sets status of specific helix position of specific turns' type to a residue.
     */
    void setStatus(HelixPositions helixPosition, turnsTypes turn);
    /*! \brief
     * Function that sets status "Break" to a residue and it's break partner.
     */
    void setBreak(secondaryStructuresData* breakPartner);
    /*! \brief
     * Function that sets status "Bridge" or "Anti-Bridge" to a residue and it's bridge partner.
     */
    void setBridge(std::size_t bridgePartnerIndex, bridgeTypes bridgeType);
    /*! \brief
     * Function that returns array of residue's bridges indexes.
     */
    std::vector<std::size_t> getBridges(bridgeTypes bridgeType);
    /*! \brief
     * Function that returns boolean status of specific secondary structure from a residue.
     */
    bool getStatus(secondaryStructureTypes secondaryStructureTypeName) const;
    /*! \brief
     * Function that returns boolean status of break existence with another specific residue.
     */
    bool isBreakPartnerWith(const secondaryStructuresData* partner) const;
    /*! \brief
     * Function that returns boolean status of bridge existence with another specific residue.
     */
    bool hasBridges(bridgeTypes bridgeType) const;
    /*! \brief
     * Function that returns helix position's status of specific turns' type in a residue.
     */
    HelixPositions getStatus(turnsTypes turn) const;
    /*! \brief
     * Function that returns status of specific secondary structure in a residue.
     */
    secondaryStructureTypes getStatus() const;

private:
    //! Boolean array of secondary structures' statuses of a residue.
    std::array<bool, static_cast<std::size_t>(secondaryStructureTypes::Count)> SecondaryStructuresStatusArray = {
        true, false, false, false, false, false, false
    };
    //! Array of pointers to other residues that forms breaks with this residue.
    secondaryStructuresData* breakPartners[2] = { nullptr, nullptr };
    //! Array of other residues indexes that forms parralel bridges with this residue.
    std::vector<std::size_t> ParallelBridgePartners;
    //! Array of other residues indexes that forms antiparallel bridges with this residue.
    std::vector<std::size_t> AntiBridgePartners;
    //! Secondary structure's status of this residue.
    secondaryStructureTypes SecondaryStructuresStatus = secondaryStructureTypes::Loop;
    //! Array of helix positions from different helix types of this residue.
    std::array<HelixPositions, static_cast<std::size_t>(turnsTypes::Count)> TurnsStatusArray{
        HelixPositions::None,
        HelixPositions::None,
        HelixPositions::None,
        HelixPositions::None
    };
};

void secondaryStructuresData::setStatus(const secondaryStructureTypes secondaryStructureTypeName, bool status)
{
    SecondaryStructuresStatusArray[static_cast<std::size_t>(secondaryStructureTypeName)] = status;
    if (status)
    {
        SecondaryStructuresStatus = secondaryStructureTypeName;
    }
    else
    {
        SecondaryStructuresStatus = secondaryStructureTypes::Loop;
    }
}

void secondaryStructuresData::setStatus(const HelixPositions helixPosition, const turnsTypes turn)
{
    TurnsStatusArray[static_cast<std::size_t>(turn)] = helixPosition;
}

bool secondaryStructuresData::getStatus(const secondaryStructureTypes secondaryStructureTypeName) const
{
    return SecondaryStructuresStatusArray[static_cast<std::size_t>(secondaryStructureTypeName)];
}

bool secondaryStructuresData::isBreakPartnerWith(const secondaryStructuresData* partner) const
{
    return breakPartners[0] == partner || breakPartners[1] == partner;
}

HelixPositions secondaryStructuresData::getStatus(const turnsTypes turn) const
{
    return TurnsStatusArray[static_cast<std::size_t>(turn)];
}

secondaryStructureTypes secondaryStructuresData::getStatus() const
{
    return SecondaryStructuresStatus;
}

void secondaryStructuresData::setBreak(secondaryStructuresData* breakPartner)
{
    if (breakPartners[0] == nullptr)
    {
        breakPartners[0] = breakPartner;
    }
    else
    {
        breakPartners[1] = breakPartner;
    }
    setStatus(secondaryStructureTypes::Break);
}

void secondaryStructuresData::setBridge(std::size_t bridgePartnerIndex, bridgeTypes bridgeType)
{
    if (bridgeType == bridgeTypes::ParallelBridge)
    {
        ParallelBridgePartners.push_back(bridgePartnerIndex);
    }
    else if (bridgeType == bridgeTypes::AntiParallelBridge)
    {
        AntiBridgePartners.push_back(bridgePartnerIndex);
    }
}

std::vector<std::size_t> secondaryStructuresData::getBridges(bridgeTypes bridgeType)
{
    if (bridgeType == bridgeTypes::ParallelBridge)
    {
        return ParallelBridgePartners;
    }
    else
    {
        return AntiBridgePartners;
    }
}

bool secondaryStructuresData::hasBridges(bridgeTypes bridgeType) const
{
    if (bridgeType == bridgeTypes::ParallelBridge)
    {
        return !ParallelBridgePartners.empty();
    }
    else
    {
        return !AntiBridgePartners.empty();
    }
}

/*! \brief
 * Class that provides search of specific h-bond patterns within residues.
 */
class secondaryStructures
{
public:
    //! Vector that contains h-bond pattern information-manipulating class for each residue in selection.
    std::vector<secondaryStructuresData> SecondaryStructuresStatusMap;
    /*! \brief
     * Function that receives data to operate and prepares to initiates h-bond patterns search.
     */
    void initiateSearch(const std::vector<ResInfo>& ResInfoMatrix, bool PiHelicesPreferencez);
    /*! \brief
     * Complex function that provides h-bond patterns search.
     */
    std::string patternSearch();
    /*! \brief
     * Class destructor that removes all saved data to prepare for new pattern-search in another frame.
     */
    ~secondaryStructures();

private:
    //! Constant float value of h-bond energy. If h-bond energy within residues is smaller than that value, then h-bond exists.
    const float HBondEnergyCutOff = -0.5;
    //! Boolean value that indicates the priority of calculating pi-helices.
    bool PiHelixPreference = false;
    //! String that contains result of dssp calculations for output.
    std::string SecondaryStructuresStringLine;
    //! Constant pointer to vector, containing ResInfo structure and defined outside of this class.
    const std::vector<ResInfo>* ResInfoMap = nullptr;
    /*! \brief
     * Function that provides a simple test if a h-bond exists within two residues of specific indices.
     */
    bool hasHBondBetween(std::size_t Donor, std::size_t Acceptor) const;
    /*! \brief
     * Function that provides a simple test if a chain break exists within two residues of specific indices.
     */
    bool NoChainBreaksBetween(std::size_t ResiA, std::size_t ResiB) const;
    /*! \brief
     * Function that calculates if bridge or anti-bridge exists within two residues of specific indices.
     */
    bridgeTypes calculateBridge(std::size_t ResiA, std::size_t ResiB) const;
    /*! \brief
     * Complex function that provides h-bond patterns search of bridges and strands. Part of patternSearch() complex function.
     */
    void analyzeBridgesAndStrandsPatterns();
    /*! \brief
     * Complex function that provides h-bond patterns search of turns and helices. Part of patternSearch() complex function.
     */
    void analyzeTurnsAndHelicesPatterns();
};

void secondaryStructures::initiateSearch(const std::vector<ResInfo>& ResInfoMatrix,
                                         const bool                  PiHelicesPreferencez)
{
    std::vector<std::size_t> temp;
    PiHelixPreference = PiHelicesPreferencez;
    ResInfoMap        = &ResInfoMatrix;
    SecondaryStructuresStatusMap.resize(ResInfoMatrix.size());
    SecondaryStructuresStringLine.resize(ResInfoMatrix.size(), '~');
}

bool secondaryStructures::hasHBondBetween(std::size_t Donor, std::size_t Acceptor) const
{
    return (((*ResInfoMap)[Donor].acceptor[0] == (*ResInfoMap)[Acceptor].info
             && (*ResInfoMap)[Donor].acceptorEnergy[0] < HBondEnergyCutOff)
            || ((*ResInfoMap)[Donor].acceptor[1] == (*ResInfoMap)[Acceptor].info
                && (*ResInfoMap)[Donor].acceptorEnergy[1] < HBondEnergyCutOff));
}

bool secondaryStructures::NoChainBreaksBetween(std::size_t ResiA, std::size_t ResiB) const
{
    if (ResiA > ResiB)
    {
        std::swap(ResiA, ResiB);
    }
    for (; ResiA != ResiB; ++ResiA)
    {
        if (SecondaryStructuresStatusMap[ResiA].isBreakPartnerWith(&SecondaryStructuresStatusMap[ResiA + 1])
            && SecondaryStructuresStatusMap[ResiA + 1].isBreakPartnerWith(
                    &SecondaryStructuresStatusMap[ResiA]))
        {
            return false;
        }
    }
    return true;
}

bridgeTypes secondaryStructures::calculateBridge(std::size_t ResiA, std::size_t ResiB) const
{
    if (ResiA < 1 || ResiB < 1 || ResiA + 1 >= ResInfoMap->size() || ResiB + 1 >= ResInfoMap->size())
    {
        return bridgeTypes::None;
    }
    if (NoChainBreaksBetween(ResiA - 1, ResiA + 1) && NoChainBreaksBetween(ResiB - 1, ResiB + 1)
        && (*ResInfoMap)[ResiA].prevResi && (*ResInfoMap)[ResiA].nextResi
        && (*ResInfoMap)[ResiB].prevResi && (*ResInfoMap)[ResiB].nextResi)
    {
        if ((hasHBondBetween(ResiA + 1, ResiB) && hasHBondBetween(ResiB, ResiA - 1))
            || (hasHBondBetween(ResiB + 1, ResiA) && hasHBondBetween(ResiA, ResiB - 1)))
        {
            return bridgeTypes::ParallelBridge;
        }
        else if ((hasHBondBetween(ResiA + 1, ResiB - 1) && hasHBondBetween(ResiB + 1, ResiA - 1))
                 || (hasHBondBetween(ResiB, ResiA) && hasHBondBetween(ResiA, ResiB)))
        {
            return bridgeTypes::AntiParallelBridge;
        }
        else
        {
            return bridgeTypes::None;
        }
    }
    return bridgeTypes::None;
}

void secondaryStructures::analyzeBridgesAndStrandsPatterns()
{
    for (std::size_t i = 1; i + 4 < SecondaryStructuresStatusMap.size(); ++i)
    {
        for (std::size_t j = i + 3; j + 1 < SecondaryStructuresStatusMap.size(); ++j)
        {
            switch (calculateBridge(i, j))
            {
                case bridgeTypes::ParallelBridge:
                {
                    SecondaryStructuresStatusMap[i].setBridge(j, bridgeTypes::ParallelBridge);
                    SecondaryStructuresStatusMap[j].setBridge(i, bridgeTypes::ParallelBridge);
                    break;
                }
                case bridgeTypes::AntiParallelBridge:
                {
                    SecondaryStructuresStatusMap[i].setBridge(j, bridgeTypes::AntiParallelBridge);
                    SecondaryStructuresStatusMap[j].setBridge(i, bridgeTypes::AntiParallelBridge);
                    break;
                }
                default: continue;
            }
        }
    }
    for (std::size_t i = 1; i + 1 < SecondaryStructuresStatusMap.size(); ++i)
    {
        for (std::size_t j = 1; j < 3; ++j)
        {
            for (const bridgeTypes& bridgeType :
                 { bridgeTypes::ParallelBridge, bridgeTypes::AntiParallelBridge })
            {
                if (SecondaryStructuresStatusMap[i].hasBridges(bridgeType)
                    && SecondaryStructuresStatusMap[i + j].hasBridges(bridgeType)
                    && (NoChainBreaksBetween(i - 1, i + 1) && NoChainBreaksBetween(i + j - 1, i + j + 1)))
                {
                    std::vector<std::size_t> i_partners =
                            SecondaryStructuresStatusMap[i].getBridges(bridgeType);
                    std::vector<std::size_t> j_partners =
                            SecondaryStructuresStatusMap[i + j].getBridges(bridgeType);
                    for (std::vector<std::size_t>::iterator i_partner = i_partners.begin();
                         i_partner != i_partners.end();
                         ++i_partner)
                    {
                        for (std::vector<std::size_t>::iterator j_partner = j_partners.begin();
                             j_partner != j_partners.end();
                             ++j_partner)
                        {
                            int delta = abs(static_cast<int>(*i_partner) - static_cast<int>(*j_partner));
                            if (delta < 6)
                            {
                                int second_strand_start = *i_partner;
                                int second_strand_end   = *j_partner;
                                if (second_strand_start > second_strand_end)
                                {
                                    std::swap(second_strand_start, second_strand_end);
                                }
                                for (std::size_t k = second_strand_start;
                                     k <= static_cast<std::size_t>(second_strand_end);
                                     ++k)
                                {
                                    SecondaryStructuresStatusMap[k].setStatus(secondaryStructureTypes::Strand);
                                }
                                for (std::size_t k = 0; k <= j; ++k)
                                {
                                    SecondaryStructuresStatusMap[i + k].setStatus(
                                            secondaryStructureTypes::Strand);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (std::size_t i = 1; i + 1 < SecondaryStructuresStatusMap.size(); ++i)
    {
        if (!SecondaryStructuresStatusMap[i].getStatus(secondaryStructureTypes::Strand)
            && (SecondaryStructuresStatusMap[i].hasBridges(bridgeTypes::ParallelBridge)
                || SecondaryStructuresStatusMap[i].hasBridges(bridgeTypes::AntiParallelBridge)))
        {
            SecondaryStructuresStatusMap[i].setStatus(secondaryStructureTypes::Bridge);
        }
    }
}

void secondaryStructures::analyzeTurnsAndHelicesPatterns()
{
    for (const turnsTypes& i : { turnsTypes::Turn_3, turnsTypes::Turn_4, turnsTypes::Turn_5 })
    {
        std::size_t stride = static_cast<std::size_t>(i) + 3;
        for (std::size_t j = 0; j + stride < SecondaryStructuresStatusMap.size(); ++j)
        {
            if (hasHBondBetween(j + stride, j) && NoChainBreaksBetween(j, j + stride))
            {
                SecondaryStructuresStatusMap[j + stride].setStatus(HelixPositions::End, i);

                for (std::size_t k = 1; k < stride; ++k)
                {
                    if (SecondaryStructuresStatusMap[j + k].getStatus(i) == HelixPositions::None)
                    {
                        SecondaryStructuresStatusMap[j + k].setStatus(HelixPositions::Middle, i);
                    }
                }
                if (SecondaryStructuresStatusMap[j].getStatus(i) == HelixPositions::End)
                {
                    SecondaryStructuresStatusMap[j].setStatus(HelixPositions::Start_AND_End, i);
                }
                else
                {
                    SecondaryStructuresStatusMap[j].setStatus(HelixPositions::Start, i);
                }
            }
        }
    }

    for (const turnsTypes& i : { turnsTypes::Turn_4, turnsTypes::Turn_3, turnsTypes::Turn_5 })
    {
        std::size_t stride = static_cast<std::size_t>(i) + 3;
        for (std::size_t j = 1; j + stride < SecondaryStructuresStatusMap.size(); ++j)
        {
            if ((SecondaryStructuresStatusMap[j - 1].getStatus(i) == HelixPositions::Start
                 || SecondaryStructuresStatusMap[j - 1].getStatus(i) == HelixPositions::Start_AND_End)
                && (SecondaryStructuresStatusMap[j].getStatus(i) == HelixPositions::Start
                    || SecondaryStructuresStatusMap[j].getStatus(i) == HelixPositions::Start_AND_End))
            {
                bool                    empty = true;
                secondaryStructureTypes Helix;
                switch (i)
                {
                    case turnsTypes::Turn_3:
                        for (std::size_t k = 0; empty && k < stride; ++k)
                        {
                            empty = SecondaryStructuresStatusMap[j + k].getStatus()
                                    <= secondaryStructureTypes::Helix_3;
                        }
                        Helix = secondaryStructureTypes::Helix_3;
                        break;
                    case turnsTypes::Turn_5:
                        for (std::size_t k = 0; empty && k < stride; ++k)
                        {
                            empty = SecondaryStructuresStatusMap[j + k].getStatus()
                                            <= secondaryStructureTypes::Helix_5
                                    || (PiHelixPreference
                                        && SecondaryStructuresStatusMap[j + k].getStatus()
                                                   == secondaryStructureTypes::Helix_4);
                        }
                        Helix = secondaryStructureTypes::Helix_5;
                        break;
                    default: Helix = secondaryStructureTypes::Helix_4; break;
                }
                if (empty || Helix == secondaryStructureTypes::Helix_4)
                {
                    for (std::size_t k = 0; k < stride; ++k)
                    {
                        SecondaryStructuresStatusMap[j + k].setStatus(Helix);
                    }
                }
            }
        }
    }
    for (std::size_t i = 1; i + 1 < SecondaryStructuresStatusMap.size(); ++i)
    {
        if (SecondaryStructuresStatusMap[i].getStatus() <= secondaryStructureTypes::Turn)
        {
            bool isTurn = false;
            for (const turnsTypes& j : { turnsTypes::Turn_3, turnsTypes::Turn_4, turnsTypes::Turn_5 })
            {
                std::size_t stride = static_cast<std::size_t>(j) + 3;
                for (std::size_t k = 1; k < stride and !isTurn; ++k)
                {
                    isTurn = (i >= k)
                             && (SecondaryStructuresStatusMap[i - k].getStatus(j) == HelixPositions::Start
                                 || SecondaryStructuresStatusMap[i - k].getStatus(j)
                                            == HelixPositions::Start_AND_End);
                }
            }
            if (isTurn)
            {
                SecondaryStructuresStatusMap[i].setStatus(secondaryStructureTypes::Turn);
            }
        }
    }
}

std::string secondaryStructures::patternSearch()
{
    analyzeBridgesAndStrandsPatterns();
    analyzeTurnsAndHelicesPatterns();
    for (auto i = static_cast<std::size_t>(secondaryStructureTypes::Bend);
         i != static_cast<std::size_t>(secondaryStructureTypes::Count);
         ++i)
    {
        for (std::size_t j = 0; j < SecondaryStructuresStatusMap.size(); ++j)
        {
            if (SecondaryStructuresStatusMap[j].getStatus(static_cast<secondaryStructureTypes>(i)))
            {
                SecondaryStructuresStringLine[j] = secondaryStructureTypeNames[i];
            }
        }
    }
    if (PiHelixPreference)
    {
        for (std::size_t j = 0; j < SecondaryStructuresStatusMap.size(); ++j)
        {
            if (SecondaryStructuresStatusMap[j].getStatus(secondaryStructureTypes::Helix_5)
                && SecondaryStructuresStatusMap[j].getStatus(secondaryStructureTypes::Helix_4))
            {
                SecondaryStructuresStringLine[j] =
                        secondaryStructureTypeNames[secondaryStructureTypes::Helix_5];
            }
        }
    }
    if (SecondaryStructuresStatusMap.size() > 1)
    {
        for (std::size_t i = 0, linefactor = 1; i + 1 < SecondaryStructuresStatusMap.size(); ++i)
        {
            if (SecondaryStructuresStatusMap[i].getStatus(secondaryStructureTypes::Break)
                && SecondaryStructuresStatusMap[i + 1].getStatus(secondaryStructureTypes::Break))
            {
                if (SecondaryStructuresStatusMap[i].isBreakPartnerWith(&SecondaryStructuresStatusMap[i + 1])
                    && SecondaryStructuresStatusMap[i + 1].isBreakPartnerWith(
                            &SecondaryStructuresStatusMap[i]))
                {
                    SecondaryStructuresStringLine.insert(
                            SecondaryStructuresStringLine.begin() + i + linefactor,
                            secondaryStructureTypeNames[secondaryStructureTypes::Break]);
                    ++linefactor;
                }
            }
        }
    }
    return SecondaryStructuresStringLine;
}

secondaryStructures::~secondaryStructures()
{
    SecondaryStructuresStatusMap.resize(0);
    SecondaryStructuresStringLine.resize(0);
}

class Dssp : public TrajectoryAnalysisModule
{
public:
    void initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings) override;
    void optionsFinished(TrajectoryAnalysisSettings* settings) override;
    void initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top) override;
    void analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata) override;
    void finishAnalysis(int nframes) override;
    void writeOutput() override;

private:
    //! Selections for DSSP output. Sets in initial options.
    Selection sel_;
    //! Boolean value for Preferring P-Helices mode. Sets in initial options.
    bool PPHelices = true;
    //! Enum value for creating hydrogen atoms mode. Very useful for structures without hydrogen atoms. Sets in initial options.
    HydrogenMode hMode = HydrogenMode::Gromacs;
    //! Enum value determines differend calculation methods for searching neighbour residues. Sets in initial options.
    NBSearchMethod NBSmode = NBSearchMethod::NBSearch;
    //! Real value that defines maximum distance from residue to its neighbour-residue.
    real cutoff_ = 0.9;
    //! Enum value that defines polyproline helix stretch. Can be only equal to 2 or 3. Sets in initial options.
    PPStreches pp_stretch = PPStreches::Default;
    //! Constant float value that determines the minimum possible distance (in Å) between two Ca atoms of amino acids of the protein,
    //! exceeding which a hydrogen bond between these two residues will be impossible.
    const float minimalCAdistance = 9.0;
    //! String value that defines output filename. Sets in initial options.
    std::string fnmDSSPOut_ = "dssp.dat";
    //! Vector of ResInfo struct that contains all important information about residues in the protein structure.
    std::vector<ResInfo> IndexMap;
    //! Class that calculates h-bond patterns in secondary structure map based on original DSSP algo.
    secondaryStructures PatternSearch;
    //! A storage that contains DSSP info from different frames.
    DsspStorage Storage;
    /*! \brief
     * Function that calculates atomic distances between atoms A and B based on atom indices.
     */
    static float CalculateAtomicDistances(const std::size_t& atomA,
                                          const std::size_t& atomB,
                                          const t_trxframe&  fr,
                                          const t_pbc*       pbc);
    /*! \brief
     * Function that calculates atomic distances between atoms A and B based on atom indices (for atom B) and atom coordinates (for atom A).
     */
    static float CalculateAtomicDistances(const rvec&        atomA,
                                          const std::size_t& atomB,
                                          const t_trxframe&  fr,
                                          const t_pbc*       pbc);
    /*! \brief
     * Function that calculates Dihedral Angles based on atom indices.
     */
    static float CalculateDihedralAngle(const int&        atomA,
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
     * Function that calculates bends and breakes in secondary structure map.
     */
    void calculateBends(const t_trxframe& fr, const t_pbc* pbc);

    /*! \brief
     * Function that Checks if H-Bond exist according to DSSP algo
     * kCouplingConstant = 27.888,  //  = 332 * 0.42 * 0.2
     * E = k * (1/rON + 1/rCH - 1/rOH - 1/rCN) where CO comes from one AA and NH from another
     * if R is in A
     * Hbond exists if E < -0.5
     */
    void calculateHBondEnergy(ResInfo* Donor, ResInfo* Acceptor, const t_trxframe& fr, const t_pbc* pbc) const;
};

float Dssp::CalculateAtomicDistances(const std::size_t& atomA,
                                     const std::size_t& atomB,
                                     const t_trxframe&  fr,
                                     const t_pbc*       pbc)
{
    gmx::RVec vectorBA = { 0, 0, 0 };
    pbc_dx(pbc, fr.x[atomA], fr.x[atomB], vectorBA.as_vec());
    return vectorBA.norm() * gmx::c_nm2A;
}

float Dssp::CalculateAtomicDistances(const rvec&        atomA,
                                     const std::size_t& atomB,
                                     const t_trxframe&  fr,
                                     const t_pbc*       pbc)
{
    gmx::RVec vectorBA = { 0, 0, 0 };
    pbc_dx(pbc, atomA, fr.x[atomB], vectorBA.as_vec());
    return vectorBA.norm() * gmx::c_nm2A;
}

float Dssp::CalculateDihedralAngle(const int&        atomA,
                                   const int&        atomB,
                                   const int&        atomC,
                                   const int&        atomD,
                                   const t_trxframe& fr,
                                   const t_pbc*      pbc)
{
    float     result                 = 360;
    float     vdist1                 = 0;
    float     vdist2                 = 0;
    gmx::RVec vectorBA               = { 0, 0, 0 };
    gmx::RVec vectorCD               = { 0, 0, 0 };
    gmx::RVec vectorCB               = { 0, 0, 0 };
    gmx::RVec vector_CBxBA           = { 0, 0, 0 };
    gmx::RVec vector_CBxCD           = { 0, 0, 0 };
    gmx::RVec vector_CBxvector_CBxCD = { 0, 0, 0 };
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
        vector_CBxBA[i] = (vectorCB[j] * vectorBA[k]) - (vectorCB[k] * vectorBA[j]);
        vector_CBxCD[i] = (vectorCB[j] * vectorCD[k]) - (vectorCB[k] * vectorCD[j]);
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
        vector_CBxvector_CBxCD[i] = (vectorCB[j] * vector_CBxCD[k]) - (vectorCB[k] * vector_CBxCD[j]);
    }
    vdist1 = (vector_CBxCD[XX] * vector_CBxCD[XX]) + (vector_CBxCD[YY] * vector_CBxCD[YY])
             + (vector_CBxCD[ZZ] * vector_CBxCD[ZZ]);
    vdist2 = (vector_CBxvector_CBxCD[XX] * vector_CBxvector_CBxCD[XX])
             + (vector_CBxvector_CBxCD[YY] * vector_CBxvector_CBxCD[YY])
             + (vector_CBxvector_CBxCD[ZZ] * vector_CBxvector_CBxCD[ZZ]);
    if (vdist1 > 0 and vdist2 > 0)
    {
        vdist1 = ((vector_CBxBA[XX] * vector_CBxCD[XX]) + (vector_CBxBA[YY] * vector_CBxCD[YY])
                  + (vector_CBxBA[ZZ] * vector_CBxCD[ZZ]))
                 / std::sqrt(vdist1);
        vdist2 = ((vector_CBxBA[XX] * vector_CBxvector_CBxCD[XX])
                  + (vector_CBxBA[YY] * vector_CBxvector_CBxCD[YY])
                  + (vector_CBxBA[ZZ] * vector_CBxvector_CBxCD[ZZ]))
                 / std::sqrt(vdist2);
        if (vdist1 != 0 or vdist2 != 0)
        {
            result = std::atan2(vdist2, vdist1) * gmx::c_rad2Deg;
        }
    }
    return result;
}

void Dssp::calculateDihedrals(const t_trxframe& fr, const t_pbc* pbc)
{
    const float        epsilon = 29;
    const float        phi_min = -75 - epsilon;
    const float        phi_max = -75 + epsilon;
    const float        psi_min = 145 - epsilon;
    const float        psi_max = 145 + epsilon;
    std::vector<float> phi(IndexMap.size(), 360);
    std::vector<float> psi(IndexMap.size(), 360);
    for (std::size_t i = 1; i + 1 < IndexMap.size(); ++i)
    {
        phi[i] = CalculateDihedralAngle(IndexMap[i - 1].getIndex(backboneAtomTypes::AtomC),
                                        IndexMap[i].getIndex(backboneAtomTypes::AtomN),
                                        IndexMap[i].getIndex(backboneAtomTypes::AtomCA),
                                        IndexMap[i].getIndex(backboneAtomTypes::AtomC),
                                        fr,
                                        pbc);
        psi[i] = CalculateDihedralAngle(IndexMap[i].getIndex(backboneAtomTypes::AtomN),
                                        IndexMap[i].getIndex(backboneAtomTypes::AtomCA),
                                        IndexMap[i].getIndex(backboneAtomTypes::AtomC),
                                        IndexMap[i + 1].getIndex(backboneAtomTypes::AtomN),
                                        fr,
                                        pbc);
    }
    for (std::size_t i = 1; i + 3 < IndexMap.size(); ++i)
    {
        switch (pp_stretch)
        {
            case PPStreches::Shortened:
            {
                if (phi_min > phi[i] or phi[i] > phi_max or phi_min > phi[i + 1] or phi[i + 1] > phi_max)
                {
                    continue;
                }

                if (psi_min > psi[i] or psi[i] > psi_max or psi_min > psi[i + 1] or psi[i + 1] > psi_max)
                {
                    continue;
                }

                switch (PatternSearch.SecondaryStructuresStatusMap[i].getStatus(turnsTypes::Turn_PP))
                {
                    case HelixPositions::None:
                        PatternSearch.SecondaryStructuresStatusMap[i].setStatus(
                                HelixPositions::Start, turnsTypes::Turn_PP);
                        break;

                    case HelixPositions::End:
                        PatternSearch.SecondaryStructuresStatusMap[i].setStatus(
                                HelixPositions::Start_AND_End, turnsTypes::Turn_PP);
                        break;

                    default: break;
                }
                PatternSearch.SecondaryStructuresStatusMap[i + 1].setStatus(HelixPositions::End,
                                                                            turnsTypes::Turn_PP);
                if (PatternSearch.SecondaryStructuresStatusMap[i].getStatus() == secondaryStructureTypes::Loop)
                {
                    PatternSearch.SecondaryStructuresStatusMap[i].setStatus(secondaryStructureTypes::Helix_PP);
                }
                if (PatternSearch.SecondaryStructuresStatusMap[i + 1].getStatus()
                    == secondaryStructureTypes::Loop)
                {
                    PatternSearch.SecondaryStructuresStatusMap[i + 1].setStatus(
                            secondaryStructureTypes::Helix_PP);
                }
                break;
            }
            case PPStreches::Default:
            {
                if (phi_min > phi[i] or phi[i] > phi_max or phi_min > phi[i + 1]
                    or phi[i + 1] > phi_max or phi_min > phi[i + 2] or phi[i + 2] > phi_max)
                {
                    continue;
                }

                if (psi_min > psi[i] or psi[i] > psi_max or psi_min > psi[i + 1]
                    or psi[i + 1] > psi_max or psi_min > psi[i + 2] or psi[i + 2] > psi_max)
                {
                    continue;
                }
                switch (PatternSearch.SecondaryStructuresStatusMap[i].getStatus(turnsTypes::Turn_PP))
                {
                    case HelixPositions::None:
                        PatternSearch.SecondaryStructuresStatusMap[i].setStatus(
                                HelixPositions::Start, turnsTypes::Turn_PP);
                        break;

                    case HelixPositions::End:
                        PatternSearch.SecondaryStructuresStatusMap[i].setStatus(
                                HelixPositions::Start_AND_End, turnsTypes::Turn_PP);
                        break;

                    default: break;
                }
                PatternSearch.SecondaryStructuresStatusMap[i + 1].setStatus(HelixPositions::Middle,
                                                                            turnsTypes::Turn_PP);
                PatternSearch.SecondaryStructuresStatusMap[i + 2].setStatus(HelixPositions::End,
                                                                            turnsTypes::Turn_PP);
                if (PatternSearch.SecondaryStructuresStatusMap[i].getStatus() == secondaryStructureTypes::Loop)
                {
                    PatternSearch.SecondaryStructuresStatusMap[i].setStatus(secondaryStructureTypes::Helix_PP);
                }
                if (PatternSearch.SecondaryStructuresStatusMap[i + 1].getStatus()
                    == secondaryStructureTypes::Loop)
                {
                    PatternSearch.SecondaryStructuresStatusMap[i + 1].setStatus(
                            secondaryStructureTypes::Helix_PP);
                }
                if (PatternSearch.SecondaryStructuresStatusMap[i + 2].getStatus()
                    == secondaryStructureTypes::Loop)
                {
                    PatternSearch.SecondaryStructuresStatusMap[i + 2].setStatus(
                            secondaryStructureTypes::Helix_PP);
                }
                break;
            }
            default: throw std::runtime_error("Unsupported stretch length");
        }
    }
}

void Dssp::calculateBends(const t_trxframe& fr, const t_pbc* pbc)
{
    const float benddegree = 70.0;
    const float maxdist    = 2.5;
    float       degree     = 0;
    float       vdist      = 0;
    float       vprod      = 0;
    gmx::RVec   caPos1{ 0, 0, 0 };
    gmx::RVec   caPos2{ 0, 0, 0 };
    for (std::size_t i = 0; i + 1 < IndexMap.size(); ++i)
    {
        if (CalculateAtomicDistances(static_cast<int>(IndexMap[i].getIndex(backboneAtomTypes::AtomC)),
                                     static_cast<int>(IndexMap[i + 1].getIndex(backboneAtomTypes::AtomN)),
                                     fr,
                                     pbc)
            > maxdist)
        {
            PatternSearch.SecondaryStructuresStatusMap[i].setBreak(
                    &PatternSearch.SecondaryStructuresStatusMap[i + 1]);
            PatternSearch.SecondaryStructuresStatusMap[i + 1].setBreak(
                    &PatternSearch.SecondaryStructuresStatusMap[i]);
        }
    }
    for (std::size_t i = 2; i + 2 < IndexMap.size(); ++i)
    {
        if (PatternSearch.SecondaryStructuresStatusMap[i - 2].isBreakPartnerWith(
                    &(PatternSearch.SecondaryStructuresStatusMap[i - 1]))
            || PatternSearch.SecondaryStructuresStatusMap[i - 1].isBreakPartnerWith(
                    &(PatternSearch.SecondaryStructuresStatusMap[i]))
            || PatternSearch.SecondaryStructuresStatusMap[i].isBreakPartnerWith(
                    &(PatternSearch.SecondaryStructuresStatusMap[i + 1]))
            || PatternSearch.SecondaryStructuresStatusMap[i + 1].isBreakPartnerWith(
                    &(PatternSearch.SecondaryStructuresStatusMap[i + 2])))
        {
            continue;
        }
        for (int j = 0; j < 3; ++j)
        {
            caPos1[j] = fr.x[IndexMap[i].getIndex(backboneAtomTypes::AtomCA)][j]
                        - fr.x[IndexMap[i - 2].getIndex(backboneAtomTypes::AtomCA)][j];
            caPos2[j] = fr.x[IndexMap[i + 2].getIndex(backboneAtomTypes::AtomCA)][j]
                        - fr.x[IndexMap[i].getIndex(backboneAtomTypes::AtomCA)][j];
        }
        vdist = (caPos1[0] * caPos2[0]) + (caPos1[1] * caPos2[1]) + (caPos1[2] * caPos2[2]);
        vprod = CalculateAtomicDistances(IndexMap[i - 2].getIndex(backboneAtomTypes::AtomCA),
                                         IndexMap[i].getIndex(backboneAtomTypes::AtomCA),
                                         fr,
                                         pbc)
                * gmx::c_angstrom / gmx::c_nano
                * CalculateAtomicDistances(IndexMap[i].getIndex(backboneAtomTypes::AtomCA),
                                           IndexMap[i + 2].getIndex(backboneAtomTypes::AtomCA),
                                           fr,
                                           pbc)
                * gmx::c_angstrom / gmx::c_nano;
        degree = std::acos(vdist / vprod) * gmx::c_rad2Deg;
        if (degree > benddegree)
        {
            PatternSearch.SecondaryStructuresStatusMap[i].setStatus(secondaryStructureTypes::Bend);
        }
    }
}

void Dssp::calculateHBondEnergy(ResInfo* Donor, ResInfo* Acceptor, const t_trxframe& fr, const t_pbc* pbc) const
{
    const float kCouplingConstant   = 27.888;
    const float minimalAtomDistance = 0.5;
    const float minEnergy           = -9.9;
    float       HbondEnergy         = 0;
    float       distanceNO          = 0;
    float       distanceHC          = 0;
    float       distanceHO          = 0;
    float       distanceNC          = 0;
    if (!(Donor->is_proline)
        && (Acceptor->getIndex(backboneAtomTypes::AtomC)
            && Acceptor->getIndex(backboneAtomTypes::AtomO) && Donor->getIndex(backboneAtomTypes::AtomN)
            && (Donor->getIndex(backboneAtomTypes::AtomH) || hMode == HydrogenMode::Dssp)))
    {
        distanceNO = CalculateAtomicDistances(Donor->getIndex(backboneAtomTypes::AtomN),
                                              Acceptor->getIndex(backboneAtomTypes::AtomO),
                                              fr,
                                              pbc);
        distanceNC = CalculateAtomicDistances(Donor->getIndex(backboneAtomTypes::AtomN),
                                              Acceptor->getIndex(backboneAtomTypes::AtomC),
                                              fr,
                                              pbc);
        if (hMode == HydrogenMode::Dssp)
        {
            if (Donor->prevResi != nullptr && Donor->prevResi->getIndex(backboneAtomTypes::AtomC)
                && Donor->prevResi->getIndex(backboneAtomTypes::AtomO))
            {
                gmx::RVec atomH  = fr.x[Donor->getIndex(backboneAtomTypes::AtomH)];
                gmx::RVec prevCO = fr.x[Donor->prevResi->getIndex(backboneAtomTypes::AtomC)];
                prevCO -= fr.x[Donor->prevResi->getIndex(backboneAtomTypes::AtomO)];
                float prevCODist =
                        CalculateAtomicDistances(Donor->prevResi->getIndex(backboneAtomTypes::AtomC),
                                                 Donor->prevResi->getIndex(backboneAtomTypes::AtomO),
                                                 fr,
                                                 pbc);
                atomH += prevCO / prevCODist;
                distanceHO = CalculateAtomicDistances(
                        atomH, Acceptor->getIndex(backboneAtomTypes::AtomO), fr, pbc);
                distanceHC = CalculateAtomicDistances(
                        atomH, Acceptor->getIndex(backboneAtomTypes::AtomC), fr, pbc);
            }
            else
            {
                distanceHO = distanceNO;
                distanceHC = distanceNC;
            }
        }
        else
        {
            distanceHO = CalculateAtomicDistances(Donor->getIndex(backboneAtomTypes::AtomH),
                                                  Acceptor->getIndex(backboneAtomTypes::AtomO),
                                                  fr,
                                                  pbc);
            distanceHC = CalculateAtomicDistances(Donor->getIndex(backboneAtomTypes::AtomH),
                                                  Acceptor->getIndex(backboneAtomTypes::AtomC),
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
    if (HbondEnergy < Donor->acceptorEnergy[0])
    {
        Donor->acceptor[1]       = Donor->acceptor[0];
        Donor->acceptorEnergy[1] = Donor->acceptorEnergy[0];
        Donor->acceptor[0]       = Acceptor->info;
        Donor->acceptorEnergy[0] = HbondEnergy;
    }
    else if (HbondEnergy < Donor->acceptorEnergy[1])
    {
        Donor->acceptor[1]       = Acceptor->info;
        Donor->acceptorEnergy[1] = HbondEnergy;
    }

    if (HbondEnergy < Acceptor->donorEnergy[0])
    {
        Acceptor->donor[1]       = Acceptor->donor[0];
        Acceptor->donorEnergy[1] = Acceptor->donorEnergy[0];
        Acceptor->donor[0]       = Donor->info;
        Acceptor->donorEnergy[0] = HbondEnergy;
    }
    else if (HbondEnergy < Acceptor->donorEnergy[1])
    {
        Acceptor->donor[1]       = Donor->info;
        Acceptor->donorEnergy[1] = HbondEnergy;
    }
}

void Dssp::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{
    static const char* const desc[] = {
        "[THISMODULE] allows using the DSSP algorithm (namely, by detecting specific patterns of "
        "hydrogen bonds between amino acid residues) "
        "to determine the secondary structure of a protein with a sufficiently high "
        "accuracy.[PAR]"
        "[TT]-hmode[tt] allows to create hydrogen pseudo-atoms based on C and O atom coordinates of"
        "previous resi.[PAR]"
        "[TT]-nb[tt] selects between two complitly diffrent ways of finding residues' pairs that"
        "will be tested on existing of hydrogen bond between them.[PAR]"
        "[TT]-cutoff[tt]. Real value that defines maximum distance from residue to its "
        "neighbour-residue."
        "Only makes sense when using with NBSearch. Recommended value is 0.9[PAR]"
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
                               .store(&hMode)
                               .defaultValue(HydrogenMode::Gromacs)
                               .enumValue(HydrogenModeNames)
                               .description("Hydrogens pseudoatoms creating mode"));
    options->addOption(EnumOption<NBSearchMethod>("nb")
                               .store(&NBSmode)
                               .defaultValue(NBSearchMethod::NBSearch)
                               .enumValue(NBSearchMethodNames)
                               .description("Method of finding neighbours of residues"));
    options->addOption(RealOption("cutoff").store(&cutoff_).required().defaultValue(0.9).description(
            "Distance form residue to its neighbour-residue in neighbour search"));
    options->addOption(
            BooleanOption("pihelix").store(&PPHelices).defaultValue(true).description("Prefer Pi Helices"));
    options->addOption(EnumOption<PPStreches>("ppstretch")
                               .store(&pp_stretch)
                               .defaultValue(PPStreches::Default)
                               .enumValue(PPStrechesNames)
                               .description("Stretch value for PP-helices"));
    settings->setHelpText(desc);
}

void Dssp::optionsFinished(TrajectoryAnalysisSettings* /* settings */) {}

void Dssp::initAnalysis(const TrajectoryAnalysisSettings& /* settings */, const TopologyInformation& top)
{
    ResInfo     _backboneAtoms;
    std::string proLINE;
    int         resicompare =
            top.atoms()->atom[static_cast<std::size_t>(*(sel_.atomIndices().begin()))].resind - 1;
    for (gmx::ArrayRef<const int>::iterator ai = sel_.atomIndices().begin();
         ai != sel_.atomIndices().end();
         ++ai)
    {
        if (resicompare != top.atoms()->atom[static_cast<std::size_t>(*ai)].resind)
        {
            resicompare = top.atoms()->atom[static_cast<std::size_t>(*ai)].resind;
            IndexMap.emplace_back(_backboneAtoms);
            IndexMap.back().info = &(top.atoms()->resinfo[resicompare]);
            proLINE              = *(IndexMap.back().info->name);
            if (proLINE == "PRO")
            {
                IndexMap[resicompare].is_proline = true;
            }
        }
        std::string atomname(*(top.atoms()->atomname[static_cast<std::size_t>(*ai)]));
        if (atomname == backboneAtomTypeNames[backboneAtomTypes::AtomCA])
        {
            IndexMap.back()._backboneIndices[static_cast<std::size_t>(backboneAtomTypes::AtomCA)] = *ai;
            IndexMap.back()._backboneIndicesStatus[static_cast<std::size_t>(backboneAtomTypes::AtomCA)] =
                    true;
        }
        else if (atomname == backboneAtomTypeNames[backboneAtomTypes::AtomC])
        {
            IndexMap.back()._backboneIndices[static_cast<std::size_t>(backboneAtomTypes::AtomC)] = *ai;
            IndexMap.back()._backboneIndicesStatus[static_cast<std::size_t>(backboneAtomTypes::AtomC)] =
                    true;
        }
        else if (atomname == backboneAtomTypeNames[backboneAtomTypes::AtomO])
        {
            IndexMap.back()._backboneIndices[static_cast<std::size_t>(backboneAtomTypes::AtomO)] = *ai;
            IndexMap.back()._backboneIndicesStatus[static_cast<std::size_t>(backboneAtomTypes::AtomO)] =
                    true;
        }
        else if (atomname == backboneAtomTypeNames[backboneAtomTypes::AtomN])
        {
            IndexMap.back()._backboneIndices[static_cast<std::size_t>(backboneAtomTypes::AtomN)] = *ai;
            IndexMap.back()._backboneIndicesStatus[static_cast<std::size_t>(backboneAtomTypes::AtomN)] =
                    true;
            if (hMode == HydrogenMode::Dssp)
            {
                IndexMap.back()._backboneIndices[static_cast<std::size_t>(backboneAtomTypes::AtomH)] = *ai;
                IndexMap.back()._backboneIndicesStatus[static_cast<std::size_t>(backboneAtomTypes::AtomH)] =
                        true;
            }
        }
        else if (hMode == HydrogenMode::Gromacs
                 && atomname == backboneAtomTypeNames[backboneAtomTypes::AtomH])
        {
            IndexMap.back()._backboneIndices[static_cast<std::size_t>(backboneAtomTypes::AtomH)] = *ai;
            IndexMap.back()._backboneIndicesStatus[static_cast<std::size_t>(backboneAtomTypes::AtomH)] =
                    true;
        }
    }
    auto isCorrupted = [](const ResInfo& Res) -> bool {
        return !Res._backboneIndicesStatus[static_cast<std::size_t>(backboneAtomTypes::AtomCA)]
               || !Res._backboneIndicesStatus[static_cast<std::size_t>(backboneAtomTypes::AtomC)]
               || !Res._backboneIndicesStatus[static_cast<std::size_t>(backboneAtomTypes::AtomO)]
               || !Res._backboneIndicesStatus[static_cast<std::size_t>(backboneAtomTypes::AtomN)]
               || !Res._backboneIndicesStatus[static_cast<std::size_t>(backboneAtomTypes::AtomH)];
    };
    auto corruptedResis = remove_if(IndexMap.begin(), IndexMap.end(), isCorrupted);
    IndexMap.erase(corruptedResis, IndexMap.end());
    for (std::size_t i = 1; i < IndexMap.size(); ++i)
    {
        IndexMap[i].prevResi     = &(IndexMap[i - 1]);
        IndexMap[i - 1].nextResi = &(IndexMap[i]);
    }
}

void Dssp::analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* /* pdata */)
{

    switch (NBSmode)
    {
        case (NBSearchMethod::NBSearch):
        {
            std::vector<gmx::RVec> positionsCA_;
            for (std::size_t i = 0; i < IndexMap.size(); ++i)
            {
                positionsCA_.emplace_back(fr.x[IndexMap[i].getIndex(backboneAtomTypes::AtomCA)]);
            }
            AnalysisNeighborhood nb_;
            nb_.setCutoff(cutoff_);
            AnalysisNeighborhoodPositions       nbPos_(positionsCA_);
            gmx::AnalysisNeighborhoodSearch     start      = nb_.initSearch(pbc, nbPos_);
            gmx::AnalysisNeighborhoodPairSearch pairSearch = start.startPairSearch(nbPos_);
            gmx::AnalysisNeighborhoodPair       pair;
            ResInfo*                            Donor;
            ResInfo*                            Acceptor;
            while (pairSearch.findNextPair(&pair))
            {
                if (pair.refIndex() < pair.testIndex())
                {
                    Donor    = &IndexMap[pair.refIndex()];
                    Acceptor = &IndexMap[pair.testIndex()];
                }
                else
                {
                    continue;
                }
                calculateHBondEnergy(Donor, Acceptor, fr, pbc);
                if (Acceptor != Donor->nextResi)
                {
                    calculateHBondEnergy(Acceptor, Donor, fr, pbc);
                }
            }
            break;
        }
        case (NBSearchMethod::Direct):
        {
            for (std::size_t Donor = 0; Donor + 1 < IndexMap.size(); ++Donor)
            {
                for (std::size_t Acceptor = Donor + 1; Acceptor < IndexMap.size(); ++Acceptor)
                {
                    if (CalculateAtomicDistances(IndexMap[Donor].getIndex(backboneAtomTypes::AtomCA),
                                                 IndexMap[Acceptor].getIndex(backboneAtomTypes::AtomCA),
                                                 fr,
                                                 pbc)
                        < minimalCAdistance)
                    {
                        calculateHBondEnergy(&IndexMap[Donor], &IndexMap[Acceptor], fr, pbc);
                        if (Acceptor != Donor + 1)
                        {
                            calculateHBondEnergy(&IndexMap[Acceptor], &IndexMap[Donor], fr, pbc);
                        }
                    }
                }
            }
            break;
        }
        default: throw std::runtime_error("Unsupported NBSearchMethod");
    }

    PatternSearch.initiateSearch(IndexMap, PPHelices);
    calculateBends(fr, pbc);
    calculateDihedrals(fr, pbc);
    Storage.addData(frnr, PatternSearch.patternSearch());
}

void Dssp::finishAnalysis(int /*nframes*/) {}

void Dssp::writeOutput()
{
    std::vector<DsspStorageFrame> dataOut_;
    FILE*                         fp_;
    fp_      = gmx_ffopen(fnmDSSPOut_, "w");
    dataOut_ = Storage.getData();
    for (auto& i : dataOut_)
    {
        std::fprintf(fp_, "%s\n", i.dsspData.c_str());
    }
    gmx_ffclose(fp_);
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
