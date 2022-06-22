#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright None- The GROMACS Authors
# and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
# Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# https://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at https://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out https://www.gromacs.org.

#! /usr/env/python3


"""Analyses code since last commit for cyclomatic complexity

Run as part of the regular testing pipeline to single out highly
complex pieces code. Depends on the lizard and pandas modules.

This code is also used for reporting code quality.

See docs/dev-manual/complexity.rst for more details.
"""

import lizard
import pandas as pd
import sys

from git import Repo

from optparse import OptionParser

def createLizardReport():
    """
    Analyses lizard data to look for highly complex code datastructures.

    Prepares data so that difference between previous and current commit can be seen.
    """
    lizardAnalysisAsListOfDicts = map(lambda x: x.__dict__, list(lizard.analyze(['.'])))
    return pd.DataFrame(lizardAnalysisAsListOfDicts)

def runAnalysisOnCurrentCheckout(commitHash):
    """
    Generate the analysis for a commit.
    """
    print("running lizard for {} (takes a few seconds)".format(commitHash))
    lizardReport = createLizardReport()
    return {'lizard-issues':lizardReport['nloc'].sum()
           }, lizardReport

def analyseAtVersion(commitHash):
    """
    Performs checkout for a version and queues the analysis.
    """
    repo = Repo('./')
    assert not repo.bare
    repo.commit(commitHash)
    return runAnalysisOnCurrentCheckout(commitHash)

def processOptions():
    """
    Handle input options.
    """

    parser = OptionParser()

    parser.add_option('--current-head', default=None,
            help='Current branch HEAD to use for checking')
    parser.add_option('--reference', default=None,
            help='Hash of commit to compare to')

    options, args = parser.parse_args()

    if not options.current_head or not options.reference:
        sys.stderr.write("Need to define both current and reference commit\n")
        sys.exit(1)

    return options


def main():
    """
    Run process of checking commits against each other.
    """

    options = processOptions()

    result_current = analyseAtVersion(options.current_head)
    result_reference = analyseAtVersion(options.reference)

    print(result_current)
    print(result_reference)

if __name__ == "__main__":
    main()


