Design and contracts
====================

:doc:`iql-overview` compares the IQL implementation with its reference
algorithms. It records the expectile and actor objectives, reference-specific
configuration choices and known gaps.

:doc:`wsrl-overview` explains the SAC, CQL, Cal-QL and WSRL layers, including
offline pretraining and the transition to online updates. Use
:doc:`/tutorials/05-offline-to-online` for the executable learning path.

:doc:`rlinf-integration` records the implemented distributed adapter designs.
It distinguishes workers that run rl-garden's own update loop from PPO's
integration with RLinf's FSDP loop and checkpoint format.

:doc:`rng-numerical-determinism` records RNG isolation for diagnostics and
floating-point reduction-order differences between critic implementations.
The existing note is in Chinese and is mounted unchanged.

:doc:`/guides/extending` is the extension contract for downstream packages.
It names registration APIs, builders and protected subclass hooks, with links
to the tested downstream fixture. Use it alongside
:doc:`/tutorials/09-extending-from-your-package`.

.. toctree::
   :maxdepth: 2

   iql-overview
   wsrl-overview
   rlinf-integration
   RNG and numerical determinism <rng-numerical-determinism>
   Extension contract <../guides/extending>
