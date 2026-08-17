"""Adapters connecting rl-garden algorithms to external training frameworks.

Each subpackage targets one external framework and imports it lazily, so
core rl_garden imports, registry discovery, and tests continue to work
without that framework installed.
"""
