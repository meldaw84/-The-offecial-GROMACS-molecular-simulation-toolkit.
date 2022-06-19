============================
Simulation input as a module
============================

Goals
=====

- Insert an appropriate layer between tools, like grompp and mdrun,
  and their input and output containers, like MDP and
  TPR files, the inputrec, etc.
- Provide a place to encapsulate input validation checks.
- Allow modernization of options expression and API access.

Note: we need to unify some handling of TPR and checkpoint data,
but it makes sense to preserve the statefulness of the simulator
(and thus the simulator input)
with respect to a new versus a continuing simulation.

grompp
======

#. Get input Builder from Simulation module.
   Optionally, get TprInputBuilder from Simulation module.
#. Get MdpDirector for MDP file from Simulation module.
#. Call director to construct input, catching errors and handling status output.
#. Alternative: iterate BuildNext calls to director for granular handling.
#. Write TPR file and/or other output.

mdrun
=====

#. Construct Simulation Input from input files.
#. Pass Input to a creation method for a new Mdrunner.
#. Mdrunner implements hook so that Simulation Input can apply checkpoint updates
   before returning Mdrunner instance.

convert-tpr
===========

#. Get TprInputBuilder from Simulation module.
#. Create TprDirector instance from existing TPR file.
#. Create EditingDirector instance for conversions.
#. TprDirector::construct()
#. EditingDirector::construct()
#. TprInputBuilder::build()
#. Alternate: insert checkpoint-merging director to write a TPR for a simulation extension.

modify_input
============

Same as convert-tpr, except that the output is an in-memory object,
plus filesystem artifact with metadata for the
operation that transformed the original TPR.

Note: ultimately, the output of the modify_input operation should have
the same fingerprint as a directly TPR-based input.
This is facilitated by having the fingerprint determined in terms of the
Simulation module rather than the modify_input operation.
This likely figures into the distinctions of the abstract workflow graph
versus the concrete execution graph.

Modularity
==========

GROMACS design favors modular compartmentalization of behaviors and data structures,
such that MDP options and checkpoint data is logically encapsulated in collective resources.
This design is not fully implemented, though.
The Simulation module can facilitate migration by taking ownership
of not-yet-modularized aspects, but allowing dispatching,
such as with extensible Input Builder and Director.

Launch schematics
=================

GROMACS 2022 Simulator launch
-----------------------------

.. uml:: simulation_launch_sequence_2022.puml

Near future updated Simulator launch
------------------------------------

.. uml:: simulation_launch_sequence_2023.puml
