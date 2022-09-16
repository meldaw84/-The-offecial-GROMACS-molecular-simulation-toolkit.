Portability
^^^^^^^^^^^

Full support for RISC-V
"""""""""""""""""""""""
We now provide full support for RISC-V, including hardware
cycle counters for efficient load balancing.

Increase of required versions
"""""""""""""""""""""""""""""
* GCC required version is now 9.
* oneMKL required version is now 2021.3.

CMake CUDA support
""""""""""""""""""
* Build system now uses ``enable_language(CUDA)`` and
  `FindCUDAToolkit <https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html>`__
  instead of `FindCUDA <https://cmake.org/cmake/help/latest/module/FindCUDA.html>`__.
  These changes may required adjustments to CMake variables for configuration CUDA
  compilation and linking.
