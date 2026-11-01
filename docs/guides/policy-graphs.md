# Policy Graphs (YAML-Declared PPO Policies)

A policy graph declares the network between the observation and the two PPO
heads as a small YAML graph: which encoders read which observation keys, which
MLP, concat or recurrent nodes sit behind them, which edges stop gradients, and
which extra nodes are trained with their own losses. rl-garden builds the
modules, infers every width, and runs the graph in the rollout and in the
update. The Gaussian actor head and the scalar value head stay the PPO contract:
they read the inputs of two sink nodes called `actor` and `value`, and are
configured by the usual PPO flags (`--net-arch`, `--log-std-init`, ...).

A graph subsumes the options that otherwise route the encoder:

- `--encoder-sharing shared_critic_grad` is a `detach: true` edge into the
  `actor` sink; `shared` is the same edge without `detach`; `separate` is two
  `encoder` nodes, one feeding each sink.
- `--obs-groups` is the `keys:` list of an `encoder` node.
- `--critic-encoder.*` is a second `encoder` node with its own `config`.

It also does what those options cannot: extra nodes (a classifier head, a
predictor) trained with a loss against a target the env reports, a skip
connection around a recurrent node, or several stateful nodes in series.

Scope: the PPO family only, see [Limits](#9-limits). Source of truth for the
format is the module docstring of `rl_garden/networks/graph/spec.py`; this guide
adds the worked examples and the behavior around checkpoints, RLinf and
extension.

## 1. Quick start

`--policy-graph` takes one of three things:

- `builtin:<name>`: a graph that reproduces a legacy policy,
- a path to a YAML file,
- inline YAML text.

Text containing a newline, `{` or `": "` is inline YAML; anything else is read
as a path (a missing or unreadable file raises `ConfigError`).

| `--policy-graph` | algorithms | trunk |
|---|---|---|
| `builtin:mlp` | `ppo` | encoder, then both heads (the `PPOPolicy` trunk) |
| `builtin:recurrent` | `recurrent_ppo`, `transformer_ppo` | encoder, then the algorithm's own sequence stage |
| `builtin:gru`, `builtin:lstm` | `recurrent_ppo` | encoder, then that stage; must equal `--rnn-type` |
| `builtin:gtrxl` | `transformer_ppo` | encoder, then a GTrXL stage |

`builtin:recurrent` resolves to `--rnn-type` (default `lstm`) under
`recurrent_ppo` and to GTrXL under `transformer_ppo`, and takes that
algorithm's stage sizes (`--rnn-hidden-size`, `--rnn-num-layers`, or the
transformer flags). An explicit `gru`, `lstm` or `gtrxl` that disagrees with the
algorithm's stage raises `ConfigError` at preflight, as does a stateful graph
under `ppo` or a stateless graph under a sequence algorithm.

```bash
# stateless: the PPOPolicy trunk
python examples/train_online.py ppo --env-id PickCube-v1 --policy-graph builtin:mlp

# recurrent: the stage follows --rnn-type (default lstm)
python examples/train_online.py recurrent_ppo --env-id PickCube-v1 \
  --policy-graph builtin:recurrent

# explicit stage, must agree with --rnn-type
python examples/train_online.py recurrent_ppo --env-id PickCube-v1 \
  --rnn-type gru --policy-graph builtin:gru

# transformer stage
python examples/train_online.py transformer_ppo --env-id PickCube-v1 \
  --policy-graph builtin:gtrxl

# your own graph, from a file
python examples/train_online.py ppo --env-id PickCube-v1 --policy-graph graphs/my_graph.yaml
```

A built-in graph reads the CLI's `--encoder.*` settings. A file or inline graph
does not: its `encoder` nodes carry their own `config`, and non-default
`--encoder.*` flags next to it raise `ConfigError`. Because the
graph decides gradient routing, `--encoder-sharing`,
`--detach-encoder-on-actor`, `--obs-groups` and `--critic-encoder.*` are
rejected next to any `--policy-graph`.

## 2. YAML reference

### Top level

```yaml
version: 1          # optional, only 1 is accepted
nodes:  {name: {type: ..., <fields>}}   # required, non-empty mapping
edges:  [{from: a, to: b, detach: false}, ...]   # required, non-empty list
losses: [...]       # optional
```

Unknown top-level fields raise `ConfigError`. Node names are Python
identifiers and must not shadow an attribute of `nn.ModuleDict` (the container
that holds the node modules): `keys`, `values`, `items`, `train`, `eval`,
`type`, `to`, `float`, `apply` and the like are rejected as reserved.

### Nodes

Exactly one `input` node, one `actor` node and one `value` node are required.

| type | fields (default) | stateful | inputs | output width |
|---|---|---|---|---|
| `input` | none | no | none | the observation (feeds `encoder` nodes only) |
| `encoder` | `keys` (`null`: every observation key; else a non-empty list of distinct keys), `config` (`{}`: `EncoderConfig` defaults; any `EncoderConfig` field) | no | exactly one edge, from the `input` node | the built encoder's `features_dim` |
| `mlp` | `net_arch` (required list of positive widths), `output_dim` (`null`: no output layer), `activation` (`relu`; `gelu`, `mish`) | no | one | `output_dim`, or the last `net_arch` width when it is `null` |
| `concat` | none | no | one or more | sum of the input widths, concatenated in producer-name order |
| `gru`, `lstm` | `hidden_size` (256), `num_layers` (1) | yes | one | `hidden_size` |
| `gtrxl` | `embed_dim` (256, even), `head_dim` (64), `num_heads` (4), `num_layers` (3), `mlp_num` (2), `memory_len` (64), `dropout_rate` (0.0, in `[0, 1)`), `gru_bias` (2.0) | yes | one | `embed_dim` |
| `actor` | none (sink: the actor head reads its input) | no | one | the input's width |
| `value` | none (sink: the value head reads its input) | no | one | the input's width |

The actor and value heads are sized from the widths of their sinks' inputs, so
the two sinks may be fed different widths (see [example c](#c-a-concat-skip-connection-into-the-value-sink)).

A graph with one or more stateful nodes (`gru`, `lstm`, `gtrxl`) runs under
`recurrent_ppo` or `transformer_ppo`; a graph without one runs under `ppo`.
With a file or inline graph, the algorithm's own sequence flags
(`--rnn-hidden-size`, the transformer flags) are not read: the node fields are
the configuration.

### Edges

| field | meaning |
|---|---|
| `from`, `to` | node names (required) |
| `detach` (`false`) | `true` cuts the gradient from this consumer into the producer. Other consumers of the same producer keep theirs. An edge out of the `input` node cannot detach. |

Each (`from`, `to`) pair may appear once. A node's incoming edges are ordered by
producer name, which is the order `concat` uses. `detach` cuts the whole latent
carried by that edge.

### Losses

Each entry trains one non-input, non-sink node with a loss that is added to the
PPO loss.

| field | meaning |
|---|---|
| `node` | the node whose output is the prediction. At most one loss per node: put a second loss on a second node. A node with a loss and no outgoing edge is valid. |
| `type` | `mse`, `bce_with_logits` (the node outputs logits) or `cross_entropy` (the node outputs class logits). More via `register_graph_loss`, see [section 8](#8-extending). |
| `target` | `info.<key>`, `batch.rewards` or `batch.dones` (below) |
| `weight` (`1.0`) | finite number `>= 0`. The loss adds `weight * mean` to the PPO loss. `0` evaluates and logs the loss but does not add it, so the parameters after an update are bit-identical to those trained without the loss term. |
| `mask` (`decision`) | `decision` or `none`. Under `--decision-level`, `decision` averages over decision steps like the policy loss and `none` over every step. Without decision-level PPO both are the plain mean. |

Targets:

- `info.<key>`: a `torch.Tensor` the env reports in `infos["<key>"]` on every
  rollout step. The info returned by the step taken from observation `o_t` is
  stored in row `t`, next to `o_t`. For `mse` and `bce_with_logits` it has shape
  `(num_envs, D)` where `D` is the node's output width (`(num_envs,)` is also
  accepted when `D == 1`) and a float or bool dtype. For `cross_entropy` it has
  shape `(num_envs,)` and an integer dtype (a class index per env). The key
  cannot be checked before the env runs: a missing key, a non-tensor, or a wrong
  shape or dtype raises `ValueError` naming the key at the first rollout step
  that shows it.
- `batch.rewards`, `batch.dones`: the step's reward and after-step done flag.
  They are one float per step, so they need a float-target loss
  (`mse`/`bce_with_logits`) on a node of width 1 (`output_dim: 1`).
  `cross_entropy` needs an `info.<key>` target.

Loss nodes that no sink depends on run only in the update, never during
rollouts. On a recurrent graph the loss node is evaluated over the whole
`(T, B)` window and flattened to `T * B` rows, T-major, exactly like its
target. The gradient of a loss reaches every upstream node unless an edge on its
path has `detach: true`. Each loss is logged as `losses/graph_loss/<node>`, the
unweighted mean over the update's minibatches (a weight-0 loss is logged too).

### The canonical form

`GraphSpec.to_yaml()` writes the canonical text: nodes and edges sorted by name,
every default spelled out (`detach`, `weight`, `mask`, node parameters,
the full `EncoderConfig` of each encoder). Two graphs that differ only in order,
comments, or omitted defaults have equal canonical text. It is what checkpoints
record and what `--print-config` shows, see [section 6](#6-checkpoints-and-evaluation).

### Validation rules and errors

Every violation raises `ConfigError` (except the ViT row) and, wherever
possible, names the node, edge or loss.

| rule | error (abridged) |
|---|---|
| duplicate YAML key | `found duplicate key '<name>'` |
| not valid YAML / not a mapping | `policy graph is not valid YAML` / `must be a mapping` |
| unknown top-level field, version other than 1 | `unknown top-level field(s)` / `version ... is not supported` |
| empty `nodes` or `edges` | `needs a non-empty 'nodes' mapping` / `'edges' list` |
| node name not an identifier, or reserved | `is not a Python identifier` / `is reserved: it is an attribute of nn.ModuleDict` |
| node without `type`, unknown `type` | `must be a mapping with a 'type' field` / `has unknown type ...; known types: [...]` |
| unknown or missing node field | `has unknown field(s) [...]` / `is missing required field '<f>'` |
| bad node field value | for example `'net_arch' must be a list of layer widths`, `'embed_dim' must be even`, `'config' has a field EncoderConfig does not know` |
| dangling edge, repeated edge, non-boolean `detach` | `names unknown node '<n>' (dangling edge)` / `is declared twice` / `'detach' must be true or false` |
| not exactly one `input`, `actor`, `value` node | `needs exactly one 'actor' node, found ...` |
| a sink with an outgoing edge | `sink node '<n>' has an outgoing edge` |
| a cycle | `has a cycle through node(s) [...]` |
| the `input` node with an incoming edge | `input node '<n>' cannot have incoming edges` |
| an `encoder` not fed by the `input` node alone | `reads the observation: it needs exactly one incoming edge, from the input node` |
| any other node fed directly by `input` | `cannot read the input node '<in>' directly; put an encoder node in between` |
| wrong input arity | `takes exactly one incoming edge, found <n>` / `has no incoming edge` (`concat`) |
| `detach` on an edge out of `input` | `an edge out of the input node ... cannot detach` |
| a dead node (no outgoing edge, no loss) | `has no outgoing edge and no loss (dead node)` |
| loss: unknown field, type, target form, mask; bad weight | `has unknown field(s)`, `has type '<t>'; known: [...]`, `has target '<t>'; expected 'info.<key>' or one of [...]`, `has mask '<m>'`, `'weight' must be a finite number >= 0` |
| loss: unknown node, or on `input`/sink | `loss on unknown node` / `loss on node '<n>': input and sink nodes produce no output of their own` |
| loss: two on one node | `node '<n>' carries more than one loss` |
| loss: `cross_entropy` on a `batch.*` target | `needs integer class targets ... use an 'info.<key>' target` |
| loss: `batch.*` target on a node wider than 1 (*) | `target '<t>' is one value per step, but the node outputs <w>; give it output_dim 1` |
| loss: one `info.<key>` used with two layouts (*) | `target 'info.<key>' is used by losses that need different layouts` |
| stateful node behind a token-layout (ViT) encoder (*) | `NotImplementedError: encoder node '<n>' feeds stateful node '<m>'` |
| stateful graph under `ppo`, or the reverse | `<Algorithm> needs a stateful/stateless policy graph` |
| `--encoder-sharing`, `--obs-groups`, ... next to a graph | `policy_graph expresses the encoder routing ... must not be set alongside it` |

`GraphSpec.parse` checks the rules without a mark; it needs no observation
space, so the CLI preflight runs it before any environment is created. The rows
marked (*) need node widths or the built encoders and fire when the policy is
built. The last two rows are checked by the CLI preflight and again when the
algorithm is constructed.

## 3. Worked examples

Examples (b) to (e) are graphs that the test suite builds into policies and runs
under the algorithm named beside each, with generic node names here. Example (a)
is the short form of the graph `builtin:mlp` builds. The node sizes are small;
scale them for real runs.

### (a) The default PPO as a graph

The graph that `builtin:mlp` expands to: one encoder feeding both heads, with
the actor edge detached so that only the value loss trains the encoder
(`shared_critic_grad`). Remove `detach: true` for `shared`.

```yaml
version: 1
nodes:
  obs:     {type: input}
  encoder: {type: encoder}
  actor:   {type: actor}
  value:   {type: value}
edges:
  - {from: obs, to: encoder}
  - {from: encoder, to: actor, detach: true}
  - {from: encoder, to: value}
```

Run with `ppo`.

### (b) A recurrent trunk with an auxiliary head and a loss

An MLP head `aux_head` reads the GRU output and is trained to predict the done
flag of each step with `bce_with_logits`. The `mask: decision` loss is averaged
over decision steps, like the policy loss, when the run uses `--decision-level`.
`aux_head` has no outgoing edge, so the loss is what keeps it in the graph; it
runs only in the update. The edge into it is not detached, so its gradient
also trains the GRU and the encoder; set `detach: true` on that edge to train
only the head.

```yaml
nodes:
  obs:      {type: input}
  encoder:  {type: encoder}
  trunk:    {type: mlp, net_arch: [8]}
  gru:      {type: gru, hidden_size: 8}
  aux_head: {type: mlp, net_arch: [8], output_dim: 1}
  actor:    {type: actor}
  value:    {type: value}
edges:
  - {from: obs, to: encoder}
  - {from: encoder, to: trunk}
  - {from: trunk, to: gru}
  - {from: gru, to: actor}
  - {from: gru, to: value}
  - {from: gru, to: aux_head}
losses:
  - {node: aux_head, type: bce_with_logits, target: batch.dones, mask: decision}
```

```bash
python examples/train_online.py recurrent_ppo --env-id <task> --decision-level \
  --policy-graph graphs/aux_head.yaml
```

`--decision-level` needs an env that reports decision steps, see
[Decision-Level PPO](decision-level-ppo.md#decision-masks-and-policy-graphs);
without the flag, `mask: decision` is the plain mean.

To train against a quantity the env reports instead, widen the head to the
target's width `D` and point the loss at an info key (here `D = 4`):

```yaml
  aux_head: {type: mlp, net_arch: [8], output_dim: 4}
# ...
losses:
  - {node: aux_head, type: mse, target: info.aux_target}
```

The env must then put a `(num_envs, 4)` float `torch.Tensor` in
`infos["aux_target"]` on every step. For a classifier use `output_dim:
<classes>`, `type: cross_entropy` and an integer `(num_envs,)` tensor.

### (c) A concat skip connection into the value sink

The actor reads the GRU; the value head reads the encoder output concatenated
with the GRU output. The two sinks have different widths, and the value head is
sized for the concat.

```yaml
nodes:
  obs:     {type: input}
  encoder: {type: encoder}
  gru:     {type: gru, hidden_size: 8}
  fused:   {type: concat}
  actor:   {type: actor}
  value:   {type: value}
edges:
  - {from: obs, to: encoder}
  - {from: encoder, to: gru}
  - {from: gru, to: actor}
  - {from: encoder, to: fused}
  - {from: gru, to: fused}
  - {from: fused, to: value}
```

Run with `recurrent_ppo`. The `concat` inputs are ordered by producer name
(`encoder`, then `gru`).

### (d) Two encoders over key subsets

The actor sees only the `state` vector; the value head also sees a camera
(a `rgb_cam` key). This is `--obs-groups` plus `--critic-encoder.*` in one
graph. The key names must exist in the env's observation.

```yaml
nodes:
  obs:            {type: input}
  actor_encoder:  {type: encoder, keys: [state]}
  critic_encoder: {type: encoder, keys: [state, rgb_cam], config: {features_dim: 16, plain_conv_pooling: gap}}
  actor:          {type: actor}
  value:          {type: value}
edges:
  - {from: obs, to: actor_encoder}
  - {from: obs, to: critic_encoder}
  - {from: actor_encoder, to: actor}
  - {from: critic_encoder, to: value}
```

Run with `ppo` on an env with an image observation.

### (e) Two stateful nodes in series

A GRU followed by a GTrXL. The policy's hidden state is one flat tuple of
tensors, the nodes' states concatenated in topological order; the rollout
buffer indexes and masks it as one.

```yaml
nodes:
  obs:     {type: input}
  encoder: {type: encoder}
  gru:     {type: gru, hidden_size: 8}
  gtrxl:   {type: gtrxl, embed_dim: 8, head_dim: 4, num_heads: 2, num_layers: 1, mlp_num: 1, memory_len: 2}
  actor:   {type: actor}
  value:   {type: value}
edges:
  - {from: obs, to: encoder}
  - {from: encoder, to: gru}
  - {from: gru, to: gtrxl}
  - {from: gtrxl, to: actor}
  - {from: gtrxl, to: value}
```

Run with `recurrent_ppo` (any stateful graph runs under a sequence algorithm;
the stage flags of that algorithm are ignored).

## 4. Semantics that differ from the legacy policies

- **Detach is whole-latent.** `detach: true` cuts the entire latent on that
  edge. `PPOPolicy` with an image `CombinedExtractor` under
  `shared_critic_grad` detaches only the image branches of the actor path; the
  proprioceptive branch still gets the actor gradient. A graph does not
  reproduce that. `builtin:mlp` on image observations therefore does not match
  `PPOPolicy`'s objective, and `ppo` warns when it builds it; on state
  observations the two agree. Write a custom graph to choose the routing
  explicitly.
- **`stop_gradient_actor` arguments are ignored.** The edges decide;
  `actor_features_detached` is always `False` for a graph policy.
- **Gradient-norm clipping agrees only up to rounding.** The parameter order of
  a graph policy is `actor`, `value_net`, then the graph nodes; the legacy
  policies put the encoder first. The global clip norm sums over parameters in
  that order, so when the encoder has parameters (image encoders) an update can
  differ from the legacy policy by floating-point rounding. With state
  observations the encoder has no parameters and the built-in graphs train
  bit-for-bit like the legacy policies.
- **Loss gradients join the global clip norm and the weight decay.** The
  weighted losses are added to the PPO loss before the single backward, and the
  clip and optimizer see one parameter set. There is one optimizer and one
  learning rate for the whole policy: no per-node learning rate.
- **Every loss is evaluated on every rank and minibatch,** so data-parallel
  ranks backpropagate through the same parameters. A `mask: decision` loss on a
  minibatch with no decision steps is a zero that is still connected to the
  graph.
- **State-dict keys differ.** Graph nodes live under `graph_nodes.<node>.`;
  checkpoints are not interchangeable with those of the legacy policies (see
  below).

## 5. Metrics

Each graph loss is logged next to the other update metrics as
`losses/graph_loss/<node>`. A graph without losses logs none.

## 6. Checkpoints and evaluation

The checkpoint metadata records the canonical graph text in
`hyperparameters["policy_graph"]` (`null` for an agent without a graph), plus
`encoder_sharing: null` and `encoder_sharing_origin: "policy_graph"`. A built-in
name is stored as its expansion, never as `builtin:...`, so a checkpoint
pins the exact graph, including the stage sizes.

On `load`, with the default `strict=True`, the text must match exactly: a
checkpoint with a graph needs an agent with the same graph, a checkpoint without
one needs an agent without one, and any difference (a `detach`, an activation, a
loss weight or mask, a node size) raises `ValueError` naming the first
differing line, even when the parameter shapes would fit. Loading with
`strict=False` skips this check.

`examples/eval_checkpoint.py` rebuilds the training arguments from the run's
`config.json` first and then overlays the checkpoint's metadata hyperparameters,
so the canonical text in the metadata wins over whatever `--policy-graph` the
run was started with, including a `builtin:` name or a file path. Evaluation
therefore needs no `--policy-graph`. A node type or loss type registered by your
own code must be registered in the evaluation process too, see
[section 8](#8-extending).

`--print-config` shows the graph as parsed: for a file or inline graph the
canonical text replaces the source and is marked runtime-derived with the reason
`canonical policy graph`; a `builtin:` name is shown as given.

## 7. RLinf

The RLinf PPO adapter (`rl_garden/integrations/rlinf/ppo_model.py`) accepts
`rlgarden_policy: graph_ppo`, with the spec as a mapping under
`rlgarden_policy_kwargs["graph"]`. The graph must be stateless (the adapter
implements the PPO contract without hidden state; a stateful graph raises
`ValueError`). The actor and rollout workers each build the policy from the same
config, so the graph mapping must be identical on both, which a single config
guarantees (an OmegaConf mapping is accepted). The observation is the flat
`obs_dim` state vector, and the tested graph uses a default `encoder` node. Other
keywords in `rlgarden_policy_kwargs` are `GraphPPOPolicy` head arguments
(`net_arch`, ...).

```yaml
actor:
  model:
    obs_dim: 6
    action_dim: 3
    rlgarden_policy: graph_ppo
    rlgarden_policy_kwargs:
      net_arch: [16]
      graph:
        nodes:
          obs:     {type: input}
          encoder: {type: encoder}
          trunk:   {type: mlp, net_arch: [8]}
          actor:   {type: actor}
          value:   {type: value}
        edges:
          - {from: obs, to: encoder}
          - {from: encoder, to: trunk}
          - {from: trunk, to: actor}
          - {from: trunk, to: value}
```

RLinf's own generic PPO losses drive the update there, not `PPO.train()`, so a
`losses:` section is not trained under RLinf: a loss-only node is never run.
Use `losses:` with the rl-garden trainers.

## 8. Extending

### New node types

```python
register_graph_node(
    name: str,
    factory: GraphNodeFactory,
    *,
    params: Optional[Mapping[str, Any]] = None,
    stateful: bool = False,
    reads_observation: bool = False,
    multi_input: bool = False,
    normalize_params: Optional[Callable[[str, dict], dict]] = None,
) -> None
```

`from rl_garden.networks.graph import REQUIRED, register_graph_node`. `params`
maps each accepted field to its default (`REQUIRED` for a mandatory one);
fields outside it are rejected, defaults are filled in the canonical form.
`normalize_params(node_name, params)` does deeper validation and may rewrite
the parameters (raise `ConfigError`). `factory(spec, input_dim,
observation_space, action_space)` returns an `nn.Module` with an integer
`features_dim` (its output width) and a `forward(x)`; `spec.params` holds the
node's parameters and `input_dim` the summed width of its inputs (`None` for a
node with `reads_observation=True`, which reads the `input` node's observation
and takes exactly one edge, from it). A `multi_input` node receives a list of
tensors. A stateful node implements the sequence-encoder protocol
(`rl_garden.networks.sequence_encoder.SequenceLatentEncoder`: `features_dim`,
`get_initial_state`, `mask_state`, `index_state`, `step`, `forward_sequence`)
instead of a plain `forward`. Registering a name twice raises `ValueError`.

```python
import torch.nn as nn

from rl_garden.networks.graph import register_graph_node


class Scale(nn.Module):
    def __init__(self, input_dim: int, factor: float) -> None:
        super().__init__()
        self.features_dim = input_dim
        self.factor = factor

    def forward(self, x):
        return self.factor * x


def build_scale(spec, input_dim, observation_space, action_space):
    return Scale(input_dim, spec.params["factor"])


register_graph_node("scale", build_scale, params={"factor": 2.0})
```

```yaml
nodes:
  obs:     {type: input}
  encoder: {type: encoder}
  twice:   {type: scale}
  actor:   {type: actor}
  value:   {type: value}
edges:
  - {from: obs, to: encoder}
  - {from: encoder, to: twice}
  - {from: twice, to: actor}
  - {from: encoder, to: value}
```

Registration has to happen before the graph is parsed, in every process that
parses it (the trainer, `eval_checkpoint`, RLinf workers). Two ways:

- A package that is installed advertises a module under the `rlgarden.graph_nodes`
  entry-point group; rl-garden imports it lazily on the first node-type lookup,
  and the module registers its node types at import time:

  ```toml
  [project.entry-points."rlgarden.graph_nodes"]
  my_nodes = "my_package.graph_nodes"
  ```

  Importing `rl_garden` never imports a node module.
- A project-local package that is not installed registers its node and loss
  types at import time, from a module that the project's own entry script
  imports before the CLI parses the graph. Env backends register the same way
  (`register_env_backend` at import, see
  [`.agents/rules/adding-env-backend.md`](../../.agents/rules/adding-env-backend.md)).

### New loss types

```python
register_graph_loss(name: str, fn: GraphLossFn, *, class_targets: bool = False)
```

`from rl_garden.networks.graph import register_graph_loss`. `fn(pred, target)`
maps the node output `pred` of shape `(rows, D)` and the target to an unreduced
loss of shape `(rows,)`; the algorithm reduces it (decision mask or plain mean)
and applies `weight`. With `class_targets=False` the target is a float tensor of
`pred`'s shape; with `class_targets=True` it is a `(rows,)` integer class index,
which only an `info.<key>` target can provide. Register before the graph is
parsed, in the same processes as for node types (the entry-point module of the
package is imported while the nodes of a graph are parsed, so it can register
loss types too).

## 9. Limits

- **PPO family only:** `ppo`, `recurrent_ppo`, `transformer_ppo`. Not SAC, DPPO,
  FlowPPO or any other algorithm: they have no `--policy-graph`. `gail` is a
  `PPO` subclass and its CLI forwards the flag, and its discriminator reads
  the `value` sink's latent; GAIL with a graph is not rejected by any check, but
  no test covers it, so it is untested.
- **The heads are not nodes.** The Gaussian actor head and the scalar value head
  are the PPO contract, built from the PPO flags; a graph decides only what
  feeds them.
- **No per-node learning rate** (one optimizer over the whole policy).
- **A token-layout (ViT `token_and_prop`) encoder cannot feed a stateful node.**
  The same limit applies to `recurrent_ppo` and `transformer_ppo` without a graph.
- **Info targets are checked at runtime,** not at preflight: a wrong key or
  shape surfaces at the first rollout step.
- **RLinf:** stateless graphs only, and `losses:` are not trained there
  (see [section 7](#7-rlinf)).
