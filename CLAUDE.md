@AGENTS.md

# Claude Code

Before training, evaluation, or remote debugging, read:
`.agents/rules/remote-training-sop.md`

Before changing or repairing Mutagen sync, read:
`.agents/rules/mutagen-sync-sop.md`

Personal server, Docker, path, Python environment, and Mutagen bindings belong
in ignored local `.agents/local/personal_config.md`, not in committed project
files. Before any remote command, read that file. If it is missing, stop and
ask the user to create it from `.agents/local/personal_config.md.example`.
