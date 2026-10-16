"""NCRL's custom dm_control tasks (cheetah/walker/quadruped/hopper).

Copied from the NCRL reference implementation (``custom_dmc_tasks``,
https://openreview.net/forum?id=oBXfPyi47m), licensed under the Apache
License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0); modified
only to resolve XML assets from this directory. Importing
this package registers every task into the upstream
``dm_control.suite.<domain>.SUITE`` containers as an import side effect, so
``dm_control.suite.load(domain, task)`` finds them. It imports
``dm_control`` -- import it only inside the env factory.
"""
from . import walker
from . import quadruped
from . import cheetah
from . import hopper
