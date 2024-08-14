################################################################################
# © Copyright 2024 Zapata Computing Inc.
################################################################################


from pyscf.tools import fcidump

from qb_gsee_benchmark.qre import get_df_qpe_circuit


def test_get_df_qpe_circuit():
    fci = fcidump.read(filename="h2_6-31g.FCIDUMP")
    circuit, n_shots, allowable_error = get_df_qpe_circuit(
        fci=fci, target_accuracy=1e-3, allowabel_failure_rate=1e-2
    )
