/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2019- The GROMACS Authors
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
/*! \file
 * \brief Exports TPR I/O tools during Python module initialization.
 *
 * Provides _gmxapi.SimulationParameters and _gmxapi.TprFile classes, as well
 * as module functions read_tprfile, write_tprfile, copy_tprfile, and rewrite_tprfile.
 *
 * TprFile is a Python object that holds a gmxapicompat::TprReadHandle.
 *
 * SimulationParameters is the Python type for data sources providing the
 * simulation parameters aspect of input to simulation operations.
 *
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 * \ingroup module_python
 */

#include "pybind11/numpy.h"

#include "gmxapi/exceptions.h"
#include "gmxapi/gmxapicompat.h"
#include "gmxapi/compat/data.h"
#include "gmxapi/compat/mdparams.h"
#include "gmxapi/compat/tpr.h"

#include "module.h"

using gmxapi::GmxapiType;
using gmxapicompat::StructureSource;

namespace py = pybind11;

namespace gmxpy
{

namespace detail
{

template<typename T>
using accessor_t = gmxapicompat::BufferDescription(const StructureSource&, const T&);

template<typename T>
py::array coordinates(const StructureSource& source, accessor_t<T>* accessor)
{
    auto buffer = accessor(source, T());
    // Expecting an Nx3 array.
    if (buffer.shape.size() != 2 || buffer.shape[1] != 3)
    {
        throw gmxapi::InternalError("Unexpected coordinates array dimensions.");
    }
    auto coordinates = py::array_t<T>({ buffer.shape[0], buffer.shape[1] },
                                      { buffer.strides[0], buffer.strides[1] });
    // Copy data from structure source until we can think some more about
    // no-copy access.
    auto proxy = coordinates.mutable_unchecked();
    for (py::ssize_t i = 0; i < proxy.shape(0); i++)
    {
        for (py::ssize_t j = 0; j < proxy.shape(1); j++)
        {
            proxy(i, j) = reinterpret_cast<T*>(buffer.ptr)[3 * i + j];
        }
    }
    return coordinates;
}

py::array positions(const StructureSource& source)
{
    if (gmxapicompat::bytesPrecision(*source.tprContents_) == sizeof(float))
    {
        return coordinates<float>(source, &gmxapicompat::positions);
    }
    else if (gmxapicompat::bytesPrecision(*source.tprContents_) == sizeof(double))
    {
        return coordinates<double>(source, &gmxapicompat::positions);
    }
    else
    {
        throw gmxapicompat::PrecisionError("Unknown floating point data type.");
    }
}


py::array velocities(const StructureSource& source)
{
    if (gmxapicompat::bytesPrecision(*source.tprContents_) == sizeof(float))
    {
        return coordinates<float>(source, &gmxapicompat::velocities);
    }
    else if (gmxapicompat::bytesPrecision(*source.tprContents_) == sizeof(double))
    {
        return coordinates<double>(source, &gmxapicompat::velocities);
    }
    else
    {
        throw gmxapicompat::PrecisionError("Unknown floating point data type.");
    }
}


void set_positions(gmxapicompat::TprWriter* writer, const py::array& coords)
{
    if (coords.ndim() != 2 || coords.shape()[1] != 3)
    {
        throw gmxapi::UsageError("Requires a Nx3 coordinates array.");
    }

    // TODO: Check type and layout to see if we can do a more optimized std::copy() or memcpy().
    // Alternatively, we could force the input to C array layout of appropriate type.
    if (coords.dtype().char_() == 'f')
    {
        const auto proxy = coords.unchecked<float, 2>();

        const std::function<float(size_t, size_t)> positions_f = [&proxy](size_t i, size_t j) {
            return proxy(i, j);
        };
        writer->positions(positions_f);
    }
    else if (coords.dtype().char_() == 'd')
    {
        const auto proxy = coords.unchecked<double, 2>();

        const std::function<double(size_t, size_t)> positions_d = [&proxy](size_t i, size_t j) {
            return proxy(i, j);
        };
        writer->positions(positions_d);
    }
    else
    {
        throw gmxapi::UsageError("Unexpected type. Expected either float32 or float64.");
    }
}

void set_velocities(gmxapicompat::TprWriter* writer, const py::array& coords)
{
    if (coords.ndim() != 2 || coords.shape()[1] != 3)
    {
        throw gmxapi::UsageError("Requires a Nx3 coordinates array.");
    }

    // TODO: Check type and layout to see if we can do a more optimized std::copy() or memcpy().
    // Alternatively, we could force the input to C array layout of appropriate type.
    if (coords.dtype().char_() == 'f')
    {
        const auto proxy = coords.unchecked<float, 2>();

        const std::function<float(size_t, size_t)> velocities_f = [&proxy](size_t i, size_t j) {
            return proxy(i, j);
        };
        writer->velocities(velocities_f);
    }
    else if (coords.dtype().char_() == 'd')
    {
        const auto proxy = coords.unchecked<double, 2>();

        const std::function<double(size_t, size_t)> velocities_d = [&proxy](size_t i, size_t j) {
            return proxy(i, j);
        };
        writer->velocities(velocities_d);
    }
    else
    {
        throw gmxapi::UsageError("Unexpected type. Expected either float32 or float64.");
    }
}

void export_tprfile(pybind11::module& module)
{
    using gmxapicompat::GmxMdParams;
    using gmxapicompat::readTprFile;
    using gmxapicompat::TprReadHandle;

    py::class_<GmxMdParams> mdparams(module, "SimulationParameters");
    // We don't want Python users to create invalid params objects, so don't
    // export a constructor until we can default initialize a valid one.
    //    mdparams.def(py::init());
    mdparams.def(
            "extract",
            [](const GmxMdParams& self) {
                py::dict dictionary;
                for (const auto& key : gmxapicompat::keys(self))
                {
                    try
                    {
                        // TODO: More complete typing and dispatching.
                        // This only handles the two types described in the initial implementation.
                        // Less trivial types (strings, maps, arrays) warrant additional
                        // design discussion before being exposed through an interface
                        // like this one.
                        // Also reference https://gitlab.com/gromacs/gromacs/-/issues/2993

                        // We can use templates and/or tag dispatch in a more complete
                        // future implementation.
                        const auto& paramType = gmxapicompat::mdParamToType(key);
                        if (paramType == GmxapiType::FLOAT64)
                        {
                            dictionary[key.c_str()] = extractParam(self, key, double());
                        }
                        else if (paramType == GmxapiType::INT64)
                        {
                            dictionary[key.c_str()] = extractParam(self, key, int64_t());
                        }
                    }
                    catch (const gmxapicompat::ValueError& e)
                    {
                        throw gmxapi::ProtocolError(std::string("Unknown parameter: ") + key);
                    }
                }
                return dictionary;
            },
            "Get a dictionary of the parameters.");

    // Overload a setter for each known type and None
    mdparams.def(
            "set",
            [](GmxMdParams* self, const std::string& key, int64_t value) {
                gmxapicompat::setParam(self, key, value);
            },
            py::arg("key").none(false),
            py::arg("value").none(false),
            "Use a dictionary to update simulation parameters.");
    mdparams.def(
            "set",
            [](GmxMdParams* self, const std::string& key, double value) {
                gmxapicompat::setParam(self, key, value);
            },
            py::arg("key").none(false),
            py::arg("value").none(false),
            "Use a dictionary to update simulation parameters.");
    mdparams.def(
            "set",
            [](GmxMdParams* self, const std::string& key, py::none) {
                // unsetParam(self, key);
                throw gmxapi::NotImplementedError(
                        "Un-setting parameters is not currently supported.");
            },
            py::arg("key").none(false),
            py::arg("value"),
            "Use a dictionary to update simulation parameters.");

    py::class_<StructureSource> structureSource(module, "SimulationStructure");
    // The two options that make the most sense are
    // 1. Create a array_t and copy the vector into it, or
    // 2. Define a type with bindings that supports the Python buffer protocol.
    // A third option might be to use a numpy structured data type...
    structureSource.def("positions", &positions, "Get a copy of the positions as a numpy array." /*,
                                       * If we use a no-copy access, we will need to extend the life
                                       of the exporting object. py::keep_alive<0, 1>()*/
    );
    structureSource.def("velocities", &velocities, "Get a copy of the velocities as a numpy array.");

    py::class_<gmxapicompat::TprWriter> tprWriter(module, "TprWriter");
    tprWriter.def(py::init(&gmxapicompat::editTprFile));
    tprWriter.def("set_positions", &set_positions);
    tprWriter.def("set_velocities", &set_velocities);
    tprWriter.def("write", &gmxapicompat::TprWriter::write);


    py::class_<TprReadHandle> tprfile(module, "TprFile");
    tprfile.def("params", [](const TprReadHandle& self) {
        auto params = gmxapicompat::getMdParams(self);
        return params;
    });
    //    tprfile.def("coordinates",
    //                [](const TprReadHandle& self)
    //                {
    //                    auto structure = gmxapicompat::getStructureSource(self);
    //                    return structure;
    //                });
    tprfile.def("positions", [](const TprReadHandle& self) {
        return positions(*gmxapicompat::getStructureSource(self));
    });
    tprfile.def("velocities", [](const TprReadHandle& self) {
        return velocities(*gmxapicompat::getStructureSource(self));
    });
    module.def("read_tprfile",
               &readTprFile,
               py::arg("filename"),
               "Get a handle to a TPR file resource for a given file name.");

    module.def(
            "write_tprfile",
            [](std::string filename, const GmxMdParams& parameterObject) {
                auto tprReadHandle = gmxapicompat::getSourceFileHandle(parameterObject);
                auto params        = gmxapicompat::getMdParams(tprReadHandle);
                auto structure     = gmxapicompat::getStructureSource(tprReadHandle);
                auto state         = gmxapicompat::getSimulationState(tprReadHandle);
                auto topology      = gmxapicompat::getTopologySource(tprReadHandle);
                gmxapicompat::writeTprFile(filename, *params, *structure, *state, *topology);
            },
            py::arg("filename").none(false),
            py::arg("parameters"),
            "Write a new TPR file with the provided data.");

    module.def(
            "copy_tprfile",
            [](const gmxapicompat::TprReadHandle& input, std::string outFile) {
                return gmxapicompat::copy_tprfile(input, outFile);
            },
            py::arg("source"),
            py::arg("destination"),
            "Copy a TPR file from ``source`` to ``destination``.");

    module.def(
            "rewrite_tprfile",
            [](std::string input, std::string output, double end_time) {
                return gmxapicompat::rewrite_tprfile(input, output, end_time);
            },
            py::arg("source"),
            py::arg("destination"),
            py::arg("end_time"),
            "Copy a TPR file from ``source`` to ``destination``, replacing `nsteps` with "
            "``end_time``.");
}

} // namespace detail

} // end namespace gmxpy
