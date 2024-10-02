################################################################################
# © Copyright 2024 Zapata Computing Inc.
################################################################################


from pyscf.tools import fcidump

from qb_gsee_benchmark.qre import get_df_qpe_circuit, get_num_shots


def test_get_num_shots():
    assert get_num_shots(square_overlap=0.9, failure_tolerance=0.1) == 1


def get_hardware_failure_tolerance_per_shot():
    assert (
        get_hardware_failure_tolerance_per_shot(failure_tolerance=0.1, num_shots=1)
        == 0.1
    )


# def test_get_df_qpe_circuit():
#     fci = fcidump.read(filename="h2_6-31g.FCIDUMP")
#     circuit, n_shots, allowable_error = get_df_qpe_circuit(
#         fci=fci, target_accuracy=1e-3, allowabel_failure_rate=1e-2
#     )
