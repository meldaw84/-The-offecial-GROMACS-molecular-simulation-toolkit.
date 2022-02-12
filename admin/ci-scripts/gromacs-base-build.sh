#!/usr/bin/env bash
set -o pipefail

CMAKE=${CMAKE:-$(which cmake)}
cd $BUILD_DIR || exit 1

# Run all the work in a sub-shell with strict error checking
(
  set -euo pipefail
  $CMAKE --build . -- -j$KUBERNETES_CPU_LIMIT 2>&1 | tee buildLogFile.log
  $CMAKE --build . --target tests -- -j$KUBERNETES_CPU_LIMIT 2>&1 | tee testBuildLogFile.log
  # Install GROMACS
  $CMAKE --build . --target install 2>&1 | tee installBuildLogFile.log
  # Remove object files to minimize artifact size
  find . -mindepth 1 -name '*.o' ! -type l -printf '%p\n' -delete &> remove-build-objects.log
)
EXITCODE=$?

# Find compiler warnings
awk '/warning/,/warning.*generated|^$/' buildLogFile.log testBuildLogFile.log \
      | grep -v "CMake" | tee buildErrors.log
grep "cannot be built" buildLogFile.log testBuildLogFile.log installBuildLogFile.log | tee -a buildErrors.log
grep "fatal error" buildLogFile.log testBuildLogFile.log | tee -a buildErrors.log
grep "error generated when compiling" buildLogFile.log testBuildLogFile.log | tee -a buildErrors.log
grep "error:" buildLogFile.log testBuildLogFile.log | tee -a buildErrors.log

# Find linking errors:
grep "^/usr/bin/ld:" buildLogFile.log testBuildLogFile.log | tee -a buildErrors.log

# Fail if there were warnings or errors reported
if [ -s buildErrors.log ] || [ $EXITCODE -ne 0 ] ; then
  echo "Found compiler warnings or errors during build"
  cat buildErrors.log
  exit 1
fi

