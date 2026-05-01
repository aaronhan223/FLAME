"""Pure-Python helpers for parsing the multi-stage ``--task`` syntax used
by the continual learning pipeline. Kept dependency-free (no torch, no
mimiciv_tasks) so the parsing logic is testable in any environment.

The continual launcher accepts ``--task`` strings in two forms:

* **Single-task-per-stage** (backward-compat): ``--task 'ihm-los-birads'``.
  Every ``-``-separated task becomes its own stage. Identical to the
  pipeline's behavior before the multi-stage extension.

* **Multi-task-per-stage**: ``--task 'ihm-los;birads'``. The ``;``
  separates stages, ``-`` enumerates joint tasks within a stage. Above
  example trains IHM + LOS jointly in stage 0, then BIRADS in stage 1.
"""


def parse_task_sequence(task_str):
    """Parse the ``--task`` string into a list of stages, each a list of
    task slugs.

    Examples:
      ``ihm-los-birads`` -> ``[['ihm'], ['los'], ['birads']]``
      ``ihm-los;birads`` -> ``[['ihm', 'los'], ['birads']]``
      ``ihm;los-birads`` -> ``[['ihm'], ['los', 'birads']]``
      ``ihm-los;birads-risk`` -> ``[['ihm', 'los'], ['birads', 'risk']]``
    """
    if ';' in task_str:
        stages = []
        for stage_str in task_str.split(';'):
            stage_str = stage_str.strip()
            if not stage_str:
                continue
            stages.append([t.strip() for t in stage_str.split('-') if t.strip()])
        return stages
    return [[t.strip()] for t in task_str.split('-') if t.strip()]


def flatten_task_arg(task_str):
    """Replace ``;`` with ``-`` so the resulting string follows the format
    ``setup_tasks_and_modalities`` already expects (a single hyphen-separated
    list of tasks).
    """
    return task_str.replace(';', '-')


def task_sequence_to_stage_indices(stages):
    """Given parsed ``stages``, return per-stage lists of *flat* task
    indices into the ``setup_tasks_and_modalities`` outputs.

    Example: ``stages = [['ihm','los'], ['birads']]`` -> ``[[0,1], [2]]``.
    """
    flat_idx = 0
    out = []
    for stage in stages:
        out.append(list(range(flat_idx, flat_idx + len(stage))))
        flat_idx += len(stage)
    return out


def path_safe_task_str(task_raw):
    """Replace ``;`` with ``__`` so the task string is safe for filesystem
    paths and shell tooling.
    """
    return task_raw.replace(';', '__')
