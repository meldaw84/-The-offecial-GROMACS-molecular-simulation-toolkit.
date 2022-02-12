#!/usr/bin/env bash
set -e
set -o pipefail
CMAKE=${CMAKE:-$(which cmake)}
cd $BUILD_DIR
$CMAKE --build . -- -j$KUBERNETES_CPU_LIMIT 2>&1 | tee buildLogFile.log
$CMAKE --build . --target tests -- -j$KUBERNETES_CPU_LIMIT 2>&1 | tee testBuildLogFile.log

# Install GROMACS
$CMAKE --build . --target install 2>&1 | tee installBuildLogFile.log

# Remove object files to minimize artifact size
find . -mindepth 1 -name '*.o' ! -type l -printf '%p\n' -delete 2>&1 > remove-build-objects.log
cd ..
