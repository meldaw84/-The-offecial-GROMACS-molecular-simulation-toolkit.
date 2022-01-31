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


