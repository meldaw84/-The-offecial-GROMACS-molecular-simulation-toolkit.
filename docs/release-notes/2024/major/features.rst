New and improved features
^^^^^^^^^^^^^^^^^^^^^^^^^

.. Note to developers!
   Please use """"""" to underline the individual entries for fixed issues in the subfolders,
   otherwise the formatting on the webpage is messed up.
   Also, please use the syntax :issue:`number` to reference issues on GitLab, without
   a space between the colon and number!


The AWH exponential histogram growth can now be controlled
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The accelerated weight histogram growth factor during the initial phase
was hard-coded to 3. Now this value can be controlled by the user.
It is set to 2 by default for increased stability.

If the TPR was generated with an earlier |Gromacs| version,
the old default value of 3 will be used.

Automatic metric scaled AWH target distribution
"""""""""""""""""""""""""""""""""""""""""""""""

The AWH target distribution can now be automatically scaled by
sqrt(AWH friction metric). Regions with higher friction (slower diffusion)
will get a higher target distribution. This can be applied to further modify
all AWH target distributions and/or AWH user input. The new option is called
'awh1-target-metric-scaling'.
