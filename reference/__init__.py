"""Reference material — contest-provided code preserved for inspiration.

The modules in this package are NOT part of the alpha-submission runtime.
They are kept here because the contest organizers wrote them and parts may
be useful to study or adapt for future strategies.

Currently included:

- :mod:`reference.contest_llm_optimizer` — the original agentic LLM-guided
  optimizer (``DCPOptimizer`` class). Reachable only via the ``--llm``
  CLI flag, which alpha submissions do not enable.

To opt back into the LLM-guided full optimizer:

    make run_optimizer DCP=... LLM=1

The deterministic dispatcher in :mod:`optimizer.dispatcher` is the alpha
default and is the path that integrates with the optional single-call
LLM augment in :mod:`optimizer.llm_augment`.
"""
