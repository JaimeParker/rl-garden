Algorithms
==========

Choose by the data you have and the update you need:

- Fresh on-policy rollouts → :doc:`ppo-family`.
- Recurrent on-policy decisions → :doc:`ppo-family`.
- Transformer on-policy decisions → :doc:`ppo-family`.
- Actions effective only at selected steps → :doc:`ppo-family` and its decision-level guide.
- YAML-declared PPO networks → :doc:`ppo-family` and its policy-graph guide.
- Online replay-based actor-critic learning → :doc:`sac-family`.
- Online learning with a fixed demonstration prior → :doc:`sac-family` and RLPD.
- Deterministic off-policy actors → :doc:`sac-family` and TD3/DrQ-v2.
- Fixed reward-labeled transitions → :doc:`offline-rl`.
- Offline flow or diffusion actor improvement → :doc:`offline-rl`.
- Offline skill representations → :doc:`offline-rl` and HILP/OPAL.
- A checkpoint or guide followed by live interaction → :doc:`offline-to-online`.
- Expert actions or a teacher to imitate → :doc:`imitation-and-generative`.
- Diffusion, flow or consistency imitation → :doc:`imitation-and-generative`.
- Learned dynamics, imagination or action planning → :doc:`model-based`.

These are navigation groups, not interchangeable policy interfaces.
Each **(phase, registry name)** belongs to one family in the generated table.
For example, offline IQL and off2on IQL are different entries with different hosts.
Decision-level PPO and policy graphs are options, not separate registrations.

Open :doc:`/reference/_generated/algorithms` for every discovered entry, its class,
family and paper reference. The table comes from the three phase registries;
an absent class thunk or unmatched reference is shown as an em dash.
The companion :doc:`/reference/_generated/backends` table lists backend classes,
dedicated installation extras and their documentation pages.

The existing deep dives remain in the site's Algorithms toctree:
:doc:`/guides/decision-level-ppo`, :doc:`/guides/policy-graphs`,
:doc:`/guides/dreamer-v3`, :doc:`/guides/ncrl`, :doc:`/guides/supermimic`,
:doc:`/guides/wsrl-reproduction`, :doc:`/guides/offline-acceleration` and
:doc:`/guides/d4rl-legacy-expansion`.
