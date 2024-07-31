"""

"""

import time

from pyscf import scf, ao2mo
from pyscf.lib import chkfile
from compute_metrics import *

from QuarticDirac import QuarticFermion


def truncate_df_eigenvalues(lambs, threshold=1e-8):
    """
    Truncate all values in the list below the specified threshold.

    Args:
    values (list of float): The list of values to truncate.
    threshold (float): The threshold below which values will be truncated.

    Returns:
    list of float: The list with values below the threshold truncated to zero.
    """
    new_list = []
    for value in lambs:
        if value >= threshold:
            new_list.append(value)
    return np.sort(new_list)
# TODO: Simplified code does not deal with spin.
#       Shouldn't be too hard to add that in,
#       we're essentially just assuming the normal
#       spin sector independence and doing the
#       analysis for all spin up


if __name__ == "__main__":

    chk = 'chks/li_cc-pVDZ_chkfile.chk'
    
    t0 = time.time()
    mol = chkfile.load_mol(chk)

    ## TO-DO: CORRECT FOR SPIN with RHF / ROHF
    mf = scf.ROHF(mol)
    scf_result_dic = chkfile.load(chk, 'scf')
    mf.__dict__.update(scf_result_dic)

    # H1 = Tne + Vnn
    hcore = mf.get_hcore()

    N_orbs = mol.nao_nr()

    # H2 = Vee
    eri_4d = ao2mo.restore(1, mol.intor('int2e'), N_orbs)
    # print(eri_4d.shape)

    t1 = time.time()
    print(f"Time to initialize data: {t1 - t0}")

    QF = QuarticFermion(eri_4d, hcore, N_orbs)
    one_body, lambs, g_mats = QF.double_factorization(purely_quartic=False)
    
    t2 = time.time()
    print(f"Time to get double factorized info: {t2 - t1}")

    MO = QF.to_majorana_operator(enumerative=True)
    # print(len(MO.data))


    t3 = time.time()
    print(f"Time to get Majorana Op: {t3 - t2}")


    MO._clear_zero_terms()
    # print(len(MO.data))

    t4 = time.time()
    print(f"Time to clear zero terms: {t4 - t3}")


    JW_Pauli_op = MO.jordan_wigner_transform(use_openfermion=False)
    """
    NOTE:   Need to implement the general transform into this package
            Can output QubitOperator object if flag is True,
            or a dictionary if flag is False.
    """

    t5 = time.time()
    print(f"Time to get JW Pauli Op: {t5 - t4}")
    
    print(f"Total time: {t5 - t0}")

    metrics = compute_hypergraph_metrics(JW_Pauli_op)

    truncated_values = truncate_df_eigenvalues(lambs)



