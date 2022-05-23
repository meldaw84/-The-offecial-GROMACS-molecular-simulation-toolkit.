The lambda dynamics module {#page_lambdaDynamics}
==============================================

`LambdaDynamics` overall organisation is described. Currently, 
`LambdaDynamics` is only used for constant pH simulations, but it can be 
further extended. Currently, `LambdaDynamics` only interpolates 
electrostatic interactions. Adding other terms might require additional 
`GROMACS` infrastructure but this is not discussed at the current stage. 

`LambdaDynamics` is based on linear interpolation of partial charges for 
atoms associated with `lambda`-coordinates. The force acting on each
`lambda`-coordinate is proportional to the electrostatic potential on
all atoms that build up the `lambda`-coordinate, computed for the system 
with charges interpolated according to the current `lambda`-values. This
requires calculation of electrostatic potentials alongside with the forces
and happens in the kernels. Because potential calculation can be used not
only by `LambdaDynamics` module, the potintials should be managed in a
general way. This is achieved with two classes: `PotentialManager` and
`PotentialEvaluator`.

Because `LambdaDynamics` can be used for various applications (e.g.
constant pH MD, redox MD...), which require specific inputs but all use
the exact same engine, `LambdaDynamics` object is kept general, while the
input treatment can be application-dependent.

The `LambdaDynamics` module is implemented using the mudular simulator.
Additional functionallity is required there to modify charges: 
`ChargeSetterManager`.

First, the overall organization of `LambdaDynamics` is discussed, next,
the initialization process is presented, and finally, the integration
into the main loop is shown.

## `LambdaDynamics`  overall organization

`LambdaDynamics` acts on `LambdaTopology` objects instead of `LambdayCoordinates` 
directly. This choice is done to ease the treatment of chemically coupled
groups (e.g. Histidine). For such groups several `lambda` groups are 
assigned, each corresponding to a physical state. These `lambda`-s are
then constrained to a (hyper-)plane on which sum of all `lambda`-s is 1.
`LambdaTopology` is a collection of individual `LambdaCoordinate` objects.
`LambdaCoordinate` includes a class for internal force calculations (
this class can differ depending on `LambdaDynamics` type. Currently only
forces are implemented). `LambdaCoordinate` also includes `LambdaStateTopology`
structure, which stores the paremeters for end states of current 
`lambda`-coordinate.

\dot
    digraph lambda_topology {
        top [label="LambdaTopology"]
        coord [label="LambdaCoordinate"]
        forces [label="LambdaInternalForces"]
        data [label="LambdaStateTopology"]

        top -> coord [label="owns"]
        coord -> data [label="owns"]
        coord -> forces [label="force getter"]
    }
\enddot

To update `LambdaTopology` object, `LambdaDynamics` uses two independent
classes: `ConstraintManager` and `UpdateManager`. `UpdateManager` takes care
of force integration and temperature coupling, while `ConstraintManager`
takes care of two types of constraints in the system:
- multi-site constraints
- charge constraints

\dot
    digraph lambda_dynamics {
        ldyn [label="LambdaDynamics"]
        pMan [label="PotentialManager"]
        cMan [label="ConstraintManager"]
        uMan [label="UpdateManager"]
        ltop [label="LambdaTopology"]
        intF [label="LambdaInternalForces"]

        ldyn -> ltop [label="owns"]
        ldyn -> cMan [label="owns"]
        ldyn -> uMan [label="owns"]
        ldyn -> pMan [label="references"]
        pMan -> uMan [label="external forces getter"]
        intF -> uMan [label="internal forces getter"]
        cMan -> uMan [label="constraint getter"]
    }
\enddot

### `UpdateManager` organization

Update manager combines all integration and temperature coupling routines.
Currently only leap-frog integration and v-rescale thermostat are 
implemented. Update can only be done if external potentials are calculated.
Thus, they are requested only when `potentialManager` receives the 
notification from `PotentialEvaluator`.

    public:
        UpdateManager(int integrator, int thermostat);
        void getExternalForces(ArrayRef<int> globalAtomIndices,
                               ArrayRef<real> chargeDifferences,
                               PotentialManager & potentials);
        void getInternalForces(ArrayRef<real> internalForces);
        std::vector<state> updateLambdas(ArrayRef<real> currentCoordinates,
                                         ArrayRef<real> currentVelocities,
                                         ConstrainManager & constraints);
    private:
        int integrator_;
        int thermostat_;
        std::vector<real> externalForces_;
        std::vector<real> internalForces_;

### `ConstraintManager` organization

Constraint manager handles all the constraints present in 
`LambdaDynamics`. At the beggining the `constraintMultipliers_` matrixt has
to be initialized once alongside with the vector of initial constraint
values `constrainVector_`, and the all `lambda`-coordinate coorections 
are computed in one step based on the current values of constraints. The 
corrections are then sent to `UpdateManager`.

    public:
        ConstraintManager(ArrayRef<int> nSites,
                          ArrayRef<real> chargeWeights,
                          ArrayRef<real> initialLambdas,
                          int numMultisiteTopologies, 
                          int numChargeConstraints);
        std::vector<real> constraintCoordinates(std::vector<real> currentLambdas);
    private:
        std::vector<int> nSites_;
        int numConstraints_;
        std::vector<real> constraintVector_;
        double** constraintMultipliers_;    
        std::vector<real> calculateCurrentConstraints(std::vector<real> currentLambdas);

## Initialization

The `LambdaDynamis` intput is stored in three files:
- `.mdp` file stores general information about `LambdaDynamics` (e.g.
masses of `lambda`-particles, reference temperature for `lambda`-coordinates,
types of `LambdaDynamics` used, etc.). Also `.mdp` stores the list of 
atom collections.
- `.ndx` files store the groups of global atom indices which correspond to
atom collections.
- `Multi-site topology` stores the parameters of `LambdaTopology` entries.

The input data is read into three input classes vecotor of `AtomCollection`,
`LambdaDynamicsParameters`, and vector of `LambdaTopologyParameters`. The
correctness of the information in these classes is first tested, and next
these classes are used to initialize `LambdaDynamics` module at both 
`grompp` and `mdrun` steps.

\dot
    digraph lambda_dynamics {
        mdp [label=".mdp"]
        ndx [label=".ndx"]
        msTop [label="Multi-site topology"]
        aa [label="Atom collection"]
        tp [label="Topology parameters"]
        ldPar [label="Lambda Dynamics parameters"]
        ld [label="LambdaDynamics"]
        pot [label="PotentialManager"]

        mdp   -> ldPar [label="initializes"]
        mdp   -> aa    [label="initializes"]
        ndx   -> aa    [label="initializes"]
        msTop -> tp    [label="initializes"]
        ldPar -> ld    [label="initializes UpdateManager"]
        tp    -> ld    [label="initializes LambdaTopology"]
        aa    -> ld    [label="initializes LambdaTopology"]
        ld    -> ld    [label="initializes ConstraintManager"]
        ld    -> pot   [label="sends list of global indices"]
        
    }
\enddot

Currently the general information of `LambdaDynamics` in `.mdp` file is 
stored in the following format:

    ; CONSTANT PH
    lambda-dynamics                                        = yes
    lambda-dynamics-calibration                            = yes
    lambda-dynamics-simulation-ph                          = 4.0
    lambda-dynamics-lambda-particle-mass                   = 5.0
    lambda-dynamics-update-nst                             = 500
    lambda-dynamics-tau                                    = 2.0
    lambda-dynamics-number-lambda-group-types              = 2
    lambda-dynamics-number-atom-collections                = 2
    lambda-dynamics-charge-constraints                     = yes

and information about atom collections is stored like this:

    lambda-dynamics-atom-set1-name                         = ASP
    lambda-dynamics-atom-set1-index-group-name             = LAMBDA1
    lambda-dynamics-atom-set1-barrier                      = 0.0        
    lambda-dynamics-atom-set1-initial-lambda               = 1.0

The format of multisite topology file specific for constant pH MD is

    [ msite_residues ]
    [ residue ]
    name    HIS_3_state
    nstates 3
    [ state1 ]
    [ atoms ]
    # AtomName AtomType Charge
    XN XT -0.5
    YN YT  0.5
    ZN ZT  0.0 
    [ end atoms ]
    [ parameters ]
    pKa 4.0
    dvdl 1 2 3 4 5
    [ end parameters ]
    [ end state1]
    [ state2 ]
    [ atoms ]
    # AtomName AtomType Charge
    XN XT1  0.5
    YN YT1  0.0
    ZN ZT1 -0.5 
    [ end atoms ]
    [ parameters ]
    pKa 4.0
    dvdl 2 3 4 5 6
    [ end parameters ]
    [ end state2]
    [ state3 ]
    [ atoms ]
    # AtomName AtomType Charge
    XN XT2  0.0
    YN YT2 -0.5
    ZN ZT2  0.5 
    [ end atoms ]
    [ parameters ]
    dvdl 2 3 1 4 5
    [ end parameters ]
    [ end state3]
    [end residue]

## Main Loop

The main loop should start by modifying the charges (or topologies 
in general). Only when new charges are assigned to atoms on all threads,
the forces and potentials can be calculated. The calculated potentials
are then grouped and sent to `LambdaDynamics`, where they are summed 
between all treads. After that `lambda`-coordinates are udpated.

\msc
hscale="2";

Topology [label="Topology"],
LambdaDynamics [label="LambdaDynamics"],
PotentialManager [label="PotentialManager"],
PotentialEvaluator [label="PotentialEvaluator"],
LambdaTopologies [label="LambdaTopologies"];

--- [ label = "New step" ];
    LambdaDynamics => Topology [ label = "Set new charges" ];
    Topology => PotentialManager [ label = "Notification: charges are ready"];
    PotentialManager => PotentialEvaluator [ label = "Request for potential calculation"];
--- [ label = "force and potentials calculation" ];
|||;
|||;
--- [ label = "forces and potentials are calculated" ];
    PotentialEvaluator => PotentialManager [ label = "Notifies that potentials are ready" ];
    LambdaDynamics => PotentialManager [ label = "Requests external forces"];
    LambdaDynamics => LambdaTopologies [ label = "Requests internal forces"];
    |||;
    LambdaDynamics => LambdaTopologies [ label = "Updates lambda coordinates" ];
--- [ label = "Write to log and edr" ];
|||;
|||;
--- [ label = "Step done" ];
\endmsc

## Future plans

In the future we plan to also interpolate Lennard-Jones interactions,
and position restraints. Also, we want to add pH-gradient and design 
inputs for redox and general `lambda`-dynamics simulations.