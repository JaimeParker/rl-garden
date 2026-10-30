"""146D observations: v4 plus effective contacts at the final physics tick."""
import numpy as np

from .state_sources import _vector
from .task_state_v4 import prepare_task_state_v4, read_task_state_v4


def prepare_task_state_v5(task):
    prepare_task_state_v4(task)
    # Reset uses the same new-scene snapshot for both contact windows.
    task._rlg_contact_last_flags_v5 = task._rlg_contact_flags_v4.copy()


def read_task_state_v5(task):
    last = _vector(task._rlg_contact_last_flags_v5, 4, 'last-tick contact flags')
    if not np.isin(last, (0., 1.)).all():
        raise ValueError('Last-tick contact flags must be binary')
    return np.concatenate((read_task_state_v4(task), last))
