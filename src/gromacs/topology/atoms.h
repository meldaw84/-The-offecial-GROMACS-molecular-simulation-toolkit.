/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 1991- The GROMACS Authors
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
#ifndef GMX_TOPOLOGY_ATOMS_H
#define GMX_TOPOLOGY_ATOMS_H

#include <stdio.h>

#include <optional>
#include <utility>
#include <vector>

#include "gromacs/topology/symtab.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/unique_cptr.h"

namespace gmx
{
class ISerializer;
} // namespace gmx

/* The particle type */
enum class ParticleType : int
{
    Atom,
    Nucleus,
    Shell,
    Bond,
    VSite,
    Count
};

/* The particle type names */
const char* enumValueToString(ParticleType enumValue);

/* Enumerated type for pdb records. The other entries are ignored
 * when reading a pdb file
 */
enum class PdbRecordType : int
{
    Atom,
    Hetatm,
    Anisou,
    Cryst1,
    Compound,
    Model,
    EndModel,
    Ter,
    Header,
    Title,
    Remark,
    Conect,
    Count
};


const char* enumValueToString(PdbRecordType enumValue);

using NameHolder = std::optional<StringTableEntry>;

//! Template wrapper struct for particle FEP state values
template<typename T>
struct FEPStateValue
{
    //! Empty object.
    FEPStateValue() : storage_({}), haveBState_(false) {}
    //! Construct without FEP state changes.
    FEPStateValue(T value) : storage_({ value, value }), haveBState_(false) {}
    //! Construct with FEP state changes.
    FEPStateValue(T valueA, T valueB) : storage_({ valueA, valueB }), haveBState_(true) {}
    //! Read from serialized datastructure.
    FEPStateValue(gmx::ISerializer* serializer);
    //! Write data to serializer.
    void serialize(gmx::ISerializer* serializer);
    //! Internal storage.
    std::array<T, 2> storage_;
    //! Whether this value has B state set or not.
    bool haveBState_;
};

using ParticleMass     = FEPStateValue<real>;
using ParticleCharge   = FEPStateValue<real>;
using ParticleTypeName = FEPStateValue<unsigned short>;

//! Single particle in a simulation.
class SimulationParticle
{
public:
    //! Write info to serializer.
    void serializeParticle(gmx::ISerializer* serializer);
    //! Access mass. A state.
    real m() const { return mass_.storage_[0]; }
    //! Access charge. A state.
    real q() const { return charge_.storage_[0]; }
    //! Access atom type. A state.
    unsigned short type() const { return atomType_.storage_[0]; }
    //! Access mass. B state.
    real mB() const { return mass_.storage_[1]; }
    //! Access charge. B state.
    real qB() const { return charge_.storage_[1]; }
    //! Access atom type. B state.
    unsigned short typeB() const { return atomType_.storage_[1]; }
    //! Access particle type.
    ParticleType ptype() const { return particleType_; }
    //! Access residue index.
    gmx::index resind() const { return residueIndex_; }
    //! Access atomic number.
    int atomnumber() const { return atomicNumber_; }
    //! Access element name.
    std::string elem() const { return elementName_; }
    //! Do we have mass?
    bool haveMass() const { return haveMass_; }
    //! Do we have charge?
    bool haveCharge() const { return haveCharge_; }
    //! Do we have type?
    bool haveType() const { return haveType_; }
    //! Do we have B state?
    bool haveBState() const { return haveBState_; }

    //! Constructor with complete information. A and B states are equivalent.
    SimulationParticle(const std::optional<ParticleMass>&     mass,
                       const std::optional<ParticleCharge>&   charge,
                       const std::optional<ParticleTypeName>& atomType,
                       ParticleType                           particleType,
                       gmx::index                             residueIndex,
                       int                                    atomicNumber,
                       const std::string&                     elementName) :
        mass_(mass.has_value() ? *mass : ParticleMass()),
        charge_(charge.has_value() ? *charge : ParticleCharge()),
        atomType_(atomType.has_value() ? *atomType : ParticleTypeName()),
        particleType_(particleType),
        residueIndex_(residueIndex),
        atomicNumber_(atomicNumber),
        elementName_(elementName),
        haveMass_(mass.has_value()),
        haveCharge_(charge.has_value()),
        haveType_(atomType.has_value()),
        haveBState_(mass_.haveBState_ && charge_.haveBState_ && atomType_.haveBState_)
    {
        GMX_ASSERT(elementName.length() <= 4, "Element name can only be three characters");
    }
    //! Construct new datastructure from deserialization.
    SimulationParticle(gmx::ISerializer* serializer);

private:
    //! Mass of the particle. A and B state.
    ParticleMass mass_;
    //! Charge of the particle. A and B state.
    ParticleCharge charge_;
    //! Atom type. A and B state.
    ParticleTypeName atomType_;
    //! Type of the particle.
    ParticleType particleType_;
    //! Residue this atoms is part of.
    gmx::index residueIndex_;
    //! Atomic Number or 0.
    int atomicNumber_;
    //! Name of the element if applicable.
    std::string elementName_;
    //! If we have mass for the particle.
    bool haveMass_;
    //! If we have charge for the particle.
    bool haveCharge_;
    //! If the particle type is set.
    bool haveType_;
    //! If all fields have B state set.
    bool haveBState_;
};

//! Single amino acid residue in a simulation
class SimulationResidue
{
public:
    //! Construct info from serializer.
    SimulationResidue(gmx::ISerializer* serializer, const StringTable& table);

    //! Construct object with complete information.
    SimulationResidue(NameHolder    name,
                      gmx::index    nr,
                      unsigned char insertionCode,
                      gmx::index    chainNumber,
                      char          chainIdentifier,
                      NameHolder    rtp) :
        name_(name),
        nr_(nr),
        insertionCode_(insertionCode),
        chainNumber_(chainNumber),
        chainIdentifier_(chainIdentifier),
        rtp_(rtp)
    {
    }
    //! Write info to serializer.
    void serializeResidue(gmx::ISerializer* serializer);
    //! Access name.
    const std::string& name() const
    {
        GMX_ASSERT(name_.has_value(), "Can not access uninitialized element");
        return *name_.value();
    }
    //! Access residue number.
    gmx::index nr() const { return nr_; }
    //! Access insertion code.
    unsigned char insertionCode() const { return insertionCode_; }
    //! Access chain number.
    gmx::index chainNumber() const { return chainNumber_; }
    //! Access chain indentifier.
    char chainIdentifier() const { return chainIdentifier_; }

private:
    //! Residue name.
    NameHolder name_;
    //! Residue number.
    gmx::index nr_;
    //! Code for insertion of residues.
    unsigned char insertionCode_;
    //! Chain number, incremented at TER or new chain identifier.
    gmx::index chainNumber_;
    //! Chain identifier written/read to pdb.
    char chainIdentifier_;
    //! Optional rtp building block name.
    NameHolder rtp_;
};

//! Defines a single line in a PDB file, for legacy PDB file handling.
class PdbEntry
{
public:
    //! Construct full structure without anisotropy information, bfactor or occupancy.
    PdbEntry(PdbRecordType type, int pdbAtomNumber, char alternativeLocation, const std::string& atomName) :
        type_(type),
        pdbAtomNumber_(pdbAtomNumber),
        alternativeLocation_(alternativeLocation),
        trueAtomName_(atomName),
        hasOccupancy_(false),
        occupancy_(0.0),
        hasbFactor_(false),
        bFactor_(0.0),
        hasAnistropy_(false)
    {
        GMX_ASSERT(atomName.length() <= 6, "Can not have atom name swith more than 6 characters");
    }
    //! Construct full structure without anisotropy information, but with bfactor and occupancy.
    PdbEntry(PdbRecordType      type,
             int                pdbAtomNumber,
             char               alternativeLocation,
             const std::string& atomName,
             real               occupancy,
             real               bFactor) :
        type_(type),
        pdbAtomNumber_(pdbAtomNumber),
        alternativeLocation_(alternativeLocation),
        trueAtomName_(atomName),
        hasOccupancy_(true),
        occupancy_(occupancy),
        hasbFactor_(true),
        bFactor_(bFactor),
        hasAnistropy_(false)
    {
        GMX_ASSERT(atomName.length() <= 6, "Can not have atom name swith more than 6 characters");
    }
    //! Construct full structure.
    PdbEntry(PdbRecordType            type,
             int                      pdbAtomNumber,
             char                     alternativeLocation,
             const std::string&       atomName,
             real                     occupancy,
             real                     bFactor,
             gmx::ArrayRef<const int> uij) :
        type_(type),
        pdbAtomNumber_(pdbAtomNumber),
        alternativeLocation_(alternativeLocation),
        trueAtomName_(atomName),
        hasOccupancy_(true),
        occupancy_(occupancy),
        hasbFactor_(true),
        bFactor_(bFactor),
        hasAnistropy_(true)
    {
        GMX_ASSERT(atomName.length() <= 6, "Can not have atom name swith more than 6 characters");
        GMX_ASSERT(uij.size() == uij_.size() || uij.empty(),
                   "Input for anisotropy needs to be exactly 6 or zero");
        int pos = 0;
        for (const auto& value : uij)
        {
            uij_[pos] = value;
            pos++;
        }
    }
    //! Get PDB record type
    PdbRecordType type() const { return type_; }
    //! Get atom number.
    int atomNumber() const { return pdbAtomNumber_; }
    //! Get access to alternative location identifier.
    char altloc() const { return alternativeLocation_; }
    //! Get access to real atom name.
    const std::string& atomName() const { return trueAtomName_; }
    //! Get access to occupancy.
    real occup() const;
    //! Get access to b factor.
    real bfac() const;
    //! Get access to anisotropy values.
    gmx::ArrayRef<const int> uij() const;

private:
    //! PDB record type
    PdbRecordType type_;
    //! PDB atom number.
    int pdbAtomNumber_;
    //! Defines alternative location in PDB.
    char alternativeLocation_;
    //! The actual atom name from the pdb file.
    std::string trueAtomName_;
    //! If entry has occupancy value.
    bool hasOccupancy_;
    //! Occupancy field, abused for other things.
    real occupancy_;
    //! If entry has bfactor value.
    bool hasbFactor_;
    //! B-Factor field, abused for other things.
    real bFactor_;
    //! If entry has anisotropy value.
    bool hasAnistropy_;
    //! Array of anisotropy values.
    std::array<int, 6> uij_;
};

/*! \brief
 * Container of finalized datastructures for Simulation atoms and residues.
 *
 * Can only be created from its builder.
 */
class SimulationMolecule
{
public:
    //! Get number of atoms.
    int numParticles() const { return particles_.size(); }
    //! Get number of residues.
    int numResidues() const { return residues_.size(); }
    //! Get number of pdbatoms.
    int numPdbAtoms() const { return pdbAtoms_.size(); }
    //! Const view on particle information.
    gmx::ArrayRef<const SimulationParticle> particles() const { return particles_; }
    //! Const view on residue information.
    gmx::ArrayRef<const SimulationResidue> residues() const { return residues_; }
    //! Const view on pdbatom information.
    gmx::ArrayRef<const PdbEntry> pdbAtoms() const { return pdbAtoms_; }


    //! If all atoms have mass.
    bool allAtomsHaveMassAndNotEmpy() const
    {
        return particles_.empty() ? false : allAtomsHaveMass_;
    }
    //! If all atoms have charge.
    bool allAtomsHaveChargeAndNotEmpty() const
    {
        return particles_.empty() ? false : allAtomsHaveCharge_;
    }
    //! If all atoms have atomnames set.
    bool allAtomsHaveAtomNameAndNotEmpty() const
    {
        return particles_.empty() ? false : allAtomsHaveAtomName_;
    }
    //! If all atoms have type information.
    bool allAtomsHaveTypeAndNotEmpty() const
    {
        return particles_.empty() ? false : allAtomsHaveType_;
    }
    //! If all atoms have b state information.
    bool allAtomsHaveBstateAndNotEmpty() const
    {
        return particles_.empty() ? false : allAtomsHaveBstate_;
    }
    //! If pdb information for all atoms has been set.
    bool allAtomsHavePdbInfoAndNotEmpty() const
    {
        return pdbAtoms_.empty() ? false : allAtomsHavePdbInfo_;
    }

    friend class SimulationMoleculeBuilder;

private:
    SimulationMolecule(std::vector<SimulationParticle>* atoms,
                       std::vector<SimulationResidue>*  residues,
                       std::vector<PdbEntry>*           pdbAtoms,
                       bool                             allAtomsHaveMass,
                       bool                             allAtomsHaveCharge,
                       bool                             allAtomsHaveAtomName,
                       bool                             allAtomsHaveType,
                       bool                             allAtomsHaveBstate,
                       bool                             allAtomsHavePdbInfo);

    //! Atom information for A state.
    std::vector<SimulationParticle> particles_;
    //! Residue information.
    std::vector<SimulationResidue> residues_;
    //! PDB information.
    std::vector<PdbEntry> pdbAtoms_;
    //! Container specific information for atom masses.
    bool allAtomsHaveMass_ = false;
    //! Container specific information for atom charges.
    bool allAtomsHaveCharge_ = false;
    //! Container specific information for atom names.
    bool allAtomsHaveAtomName_ = false;
    //! Container specific information for atom types.
    bool allAtomsHaveType_ = false;
    //! Container specific information for atom b state.
    bool allAtomsHaveBstate_ = false;
    //! Container specific information for pdbatom information.
    bool allAtomsHavePdbInfo_ = false;
};

/*! \brief \libinternal
 * Convenience class combining the new datastructures for atoms and residues.
 *
 * Used to build the final molecule by adding particles, residues or pdb entries.
 * The data used to run a simulation can be obtained from this builder.
 */
class SimulationMoleculeBuilder
{
public:
    //! Function to add new atom.
    void addParticle(const SimulationParticle& atom);
    //! Function to add new residue.
    void addResidue(const SimulationResidue& residue);
    //! Function to add new pdbatom.
    void addPdbatom(const PdbEntry& pdbatom);
    /*! \brief
     * Finalize datastructure to store information about validity of the entries.
     *
     * \returns A simulation ready datastructure and a clean state of the builder.
     */
    SimulationMolecule finalize();

    //! Get number of atoms.
    int numParticles() const { return particles_.size(); }
    //! Get number of residues.
    int numResidues() const { return residues_.size(); }
    //! Get number of pdbatoms.
    int numPdbAtoms() const { return pdbAtoms_.size(); }
    //! Const view on particles information.
    gmx::ArrayRef<const SimulationParticle> particles() const { return particles_; }
    //! Const view on residue information.
    gmx::ArrayRef<const SimulationResidue> residues() const { return residues_; }
    //! Const view on pdbatom information.
    gmx::ArrayRef<const PdbEntry> pdbAtoms() const { return pdbAtoms_; }
    //! View on particles information.
    gmx::ArrayRef<SimulationParticle> particles() { return particles_; }
    //! View on residue information.
    gmx::ArrayRef<SimulationResidue> residues() { return residues_; }
    //! View on pdbatom information.
    gmx::ArrayRef<PdbEntry> pdbAtoms() { return pdbAtoms_; }

private:
    //! Atom information for state A.
    std::vector<SimulationParticle> particles_;
    //! Residue information.
    std::vector<SimulationResidue> residues_;
    //! PDB information.
    std::vector<PdbEntry> pdbAtoms_;
};

// Legacy datastructures begin below.
typedef struct t_atom
{
    real           m, q;       /* Mass and charge                      */
    real           mB, qB;     /* Mass and charge for Free Energy calc */
    unsigned short type;       /* Atom type                            */
    unsigned short typeB;      /* Atom type for Free Energy calc       */
    ParticleType   ptype;      /* Particle type                        */
    int            resind;     /* Index into resinfo (in t_atoms)      */
    int            atomnumber; /* Atomic Number or 0                   */
    char           elem[4];    /* Element name                         */
} t_atom;

typedef struct t_resinfo
{
    char**        name;     /* Pointer to the residue name          */
    int           nr;       /* Residue number                       */
    unsigned char ic;       /* Code for insertion of residues       */
    int           chainnum; /* Iincremented at TER or new chain id  */
    char          chainid;  /* Chain identifier written/read to pdb */
    char**        rtp;      /* rtp building block name (optional)   */
} t_resinfo;

typedef struct t_pdbinfo
{
    PdbRecordType type;         /* PDB record name                      */
    int           atomnr;       /* PDB atom number                      */
    char          altloc;       /* Alternate location indicator         */
    char          atomnm[6];    /* True atom name including leading spaces */
    real          occup;        /* Occupancy                            */
    real          bfac;         /* B-factor                             */
    bool          bAnisotropic; /* (an)isotropic switch                 */
    int           uij[6];       /* Anisotropic B-factor                 */
} t_pdbinfo;

//! Contains indices into group names for different groups.
using AtomGroupIndices = std::vector<int>;

typedef struct t_atoms
{
    int     nr;         /* Nr of atoms                          */
    t_atom* atom;       /* Array of atoms (dim: nr)             */
                        /* The following entries will not       */
                        /* always be used (nres==0)             */
    char*** atomname;   /* Array of pointers to atom name       */
                        /* use: (*(atomname[i]))                */
    char*** atomtype;   /* Array of pointers to atom types      */
                        /* use: (*(atomtype[i]))                */
    char*** atomtypeB;  /* Array of pointers to B atom types    */
                        /* use: (*(atomtypeB[i]))               */
    int        nres;    /* The number of resinfo entries        */
    t_resinfo* resinfo; /* Array of residue names and numbers   */
    t_pdbinfo* pdbinfo; /* PDB Information, such as aniso. Bfac */

    /* Flags that tell if properties are set for all nr atoms.
     * For B-state parameters, both haveBState and the mass/charge/type
     * flag should be TRUE.
     */
    bool haveMass;    /* Mass available                       */
    bool haveCharge;  /* Charge available                     */
    bool haveType;    /* Atom type available                  */
    bool haveBState;  /* B-state parameters available         */
    bool havePdbInfo; /* pdbinfo available                    */
} t_atoms;

#define PERTURBED(a) (((a).mB != (a).m) || ((a).qB != (a).q) || ((a).typeB != (a).type))

void init_atom(t_atoms* at);
void done_atom(t_atoms* at);
void done_and_delete_atoms(t_atoms* atoms);

void init_t_atoms(t_atoms* atoms, int natoms, bool bPdbinfo);
/* allocate memory for the arrays, set nr to natoms and nres to 0
 * set pdbinfo to NULL or allocate memory for it */

void gmx_pdbinfo_init_default(t_pdbinfo* pdbinfo);

t_atoms* copy_t_atoms(const t_atoms* src);
/* copy an atoms struct from src to a new one */

void add_t_atoms(t_atoms* atoms, int natom_extra, int nres_extra);
/* allocate extra space for more atoms and or residues */

void t_atoms_set_resinfo(t_atoms*         atoms,
                         int              atom_ind,
                         struct t_symtab* symtab,
                         const char*      resname,
                         int              resnr,
                         unsigned char    ic,
                         int              chainnum,
                         char             chainid);
/* Set the residue name, number, insertion code and chain identifier
 * of atom index atom_ind.
 */

void pr_atoms(FILE* fp, int indent, const char* title, const t_atoms* atoms, bool bShownumbers);

/*! \brief Compare information in the t_atoms data structure.
 *
 * \param[in] fp Pointer to file to write to.
 * \param[in] a1 Pointer to first data structure to compare.
 * \param[in] a2 Pointer to second data structure or nullptr.
 * \param[in] relativeTolerance Relative floating point comparison tolerance.
 * \param[in] absoluteTolerance Absolute floating point comparison tolerance.
 */
void compareAtoms(FILE* fp, const t_atoms* a1, const t_atoms* a2, real relativeTolerance, real absoluteTolerance);

/*! \brief Set mass for each atom using the atom and residue names using a database
 *
 * If atoms->haveMass = TRUE does nothing.
 * If printMissingMasss = TRUE, prints details for first 10 missing masses
 * to stderr.
 */
void atomsSetMassesBasedOnNames(t_atoms* atoms, bool printMissingMasses);

//! Deleter for t_atoms, needed until it has a proper destructor.
using AtomsDataPtr = gmx::unique_cptr<t_atoms, done_and_delete_atoms>;


#endif
