======================
GROMACS public C++ API
======================

GROMACS public C++ API. Library and headers are installed automatically, when possible, unless the CMake
project is configured with ``-DGMXAPI=OFF``.

Building and linking
--------------------

TODO: Import and update GROMACS-as-a-library documentation from old doxygen "public API" page.

.. see https://breathe.readthedocs.io/en/latest/directives.html for Sphinx extension syntax.

Initializing the library
------------------------

.. cpp:function:: LibraryContext init(ResourceAllocation)

    Get a resource context for an API session.

.. cpp:class:: LibraryContext

    RAII management for resources allocated to the library API session. It is the client's responsibility
    to keep the object alive while resources are in use, such as MPI communicator, GPU device context, and
    thread pools. Resources are released when the object is destroyed.

Runtime I/O and environment details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The library currently includes concepts like "output environment" and "program context" that should be
wrapped into :cpp:class:`LibraryContext` or a related abstraction. However, they are currently very
commandline-centric and could use some updates.

These would also be sensible insertion points for details related to filesystem abstractions or filesystem
details, such as

* an effective working directory independent of ``CWD``
* normalization of user input from environment variables, command line arguments, etc.
* abstraction or overriding of ``stdio`` file descriptors
* installation/deinstallation of IPC signal handlers
* management of the root logger

Preparing simulation input
--------------------------

.. cpp:class:: SimulationInputBuilder

    An API-friendly, object-oriented factoring of ``grompp``.

    Allow error checking and client-side error catching to be broken up into more phases.

    Allow more flexibility in composing input from filesystem objects or other API calls.

    Provide an intermediate for more complex ``modifyInput`` use cases, or in a pipeline
    in which SimulationOutput is used to construct SimulationInput.

.. cpp:function:: SimulationInput simulationInputFromFiles(const LibraryContext&, char*... filename)

    Acquire a SimulationInput handle from filesystem data.

.. cpp:function:: SimulationInput modifyInput(const SimulationInput& src, Update update)

    Apply a modification to the referenced SimulationInput. Returns a distinct handle that
    refers to data with updates applied. Data locality of the returned handle should be
    assumed to be the same as that of the source. Actual data locality will be resolved
    when the SimulationInput is consumed, such as by a file writer or a Simulator.

.. cpp:class:: Update

    Reference "Command pattern".

.. cpp:function:: Result* simulationResult(const SimulationOutput&)

    Describe the result of the call to :cpp:expr:`simulator()`. Determine whether the trajectory
    was produced as prescribed, whether errors occurred, and whether there were any reasons for
    the call to return before the work could be completed as prescribed in the SimulationInput.

    In the initial API release, this is probably an opaque object. It may take some time to decide
    how to represent and expose Result information.

.. cpp:class:: SimulationContext

    SimulationContext is a scoped substate of LibraryContext, configured with simulation component
    code objects (in the MDModules container), and computational resources locked for an immediately
    pending simulation task.

Running Simulations
-------------------

.. cpp:function:: SimulatorBuilder simulatorBuilder(const SimulationContext&, const SimulationInput&)

.. cpp:concept:: template<typename S> Simulator

    A GROMACS Simulator is a callable object that is ready to perform computation
    to produce a trajectory for a molecular system. Input has been provided and
    resources have been allocated. It allows no mutating access except for the
    :cpp:func:`operator()()`, which starts the computation.

    Simulator objects are acquired with :cpp:func:`SimulatorBuilder::build`.

    .. cpp:var:: S simulator

        A function object.

    :cpp:expr:`simulator()` produces a :cpp:class:`SimulationOutput` when called.
    :cpp:expr:`simulator()` must be called on all participating ranks in an MPI-accelerated simulation.
    *(thread-MPI call pattern TBD)*

.. cpp:class:: SimulationInput

    Handle to complete :cpp:concept:`Simulator` input. Includes molecular model and a description of the
    prescribed computation.
    In MPI use cases, the same operations should be applied to the handle on all participating ranks.

    *TBD: data localization / distribution helpers.*

.. cpp:class:: SimulationOutput

    Handle to the :cpp:concept:`Simulator` output.

    In MPI use cases, a SimulationOutput handle exists on all ranks that participated in the simulation,
    and subsequent access must occur the same on all ranks.
    However, the handle is symbolic, and may not refer to local data on any particular rank.

    *TBD: data localization helpers.*

.. cpp:class:: SimulatorBuilder

    Acquire a SimulatorBuilder with the parameterized factory function :cpp:func:`simulatorBuilder`.

    .. cpp:function:: SimulatorBuilder& add(Component&&)

    .. cpp:function:: Simulator build()

Controlling Resource Allocation
-------------------------------

.. cpp:class:: ResourceAllocation

    Handles or parameters for computing resources allocated to the library API session.

Using the Trajectory Analysis Framework
---------------------------------------

Initializing and launching the TAF Runner.

Building the tool pipeline.

Generic interfaces of TAF modules and data.

Extension interfaces
--------------------

Register modules (names, roles, and factory functions) with the LibraryContext (or related object) for use
by future API calls (or launched work).

Document the interfaces available to or required from extension code.

Interacting with GROMACS-native file formats
--------------------------------------------

* topology
* structure
* energy log
* other logged data (xvg, etc)
* TNG trajectory output
* simulation input / run input record, augmented by checkpoint

1. API version of ``dump``
2. Builder/writer and/or editing interfaces

Data exchange
-------------

DLPack compatible data descriptors for array data.

Supported scalar types recognized for array data.

Ownership / lifetime semantics for reference objects.

Helpers: iterators, adapters, etc.

Serialization support.

Describe conventions for higher-level structured data, such as Topology and force field.

Describe data locality details, such as for mapping between a given array of atom data and the
original input.

User interface helpers
----------------------

Functions and data conventions for

* identifying tool inputs and outputs,
* extracting and formatting help text,
* getting suggested short and long option names or data keys
