from typing import Callable, Iterable, Optional

import numpy as np
from openfermion.resource_estimates import df
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyscf.tools import fcidump


def get_dmrg_energy(fci: dict, point_group: str) -> float:

    bond_dims = [250] * 4 + [500] * 4
    noises = [1e-4] * 4 + [1e-5] * 4 + [0]
    thrds = [1e-10] * 8

    fcidump_path = ".fcidump"
    fcidump.from_integrals(
        fcidump_path,
        h1e=fci["H1"],
        h2e=fci["H2"],
        nmo=fci["NORB"],
        nelec=fci["NELEC"],
        ms=fci["MS"],
    )

    driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=4)

    # read integrals from file
    driver.read_fcidump(filename=fcidump_path, pg=point_group)
    driver.initialize_system(
        n_sites=driver.n_sites,
        n_elec=driver.n_elec,
        spin=driver.spin,
        orb_sym=driver.orb_sym,
    )

    mpo = driver.get_qc_mpo(
        h1e=driver.h1e, g2e=driver.g2e, ecore=driver.ecore, iprint=1
    )
    ket = driver.get_random_mps(tag="GS", bond_dim=250, nroots=1)
    energy = driver.dmrg(
        mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises, thrds=thrds, iprint=1
    )
    return energy


def choose_double_factorization_threshold(
    fci: dict,
    candidate_thresholds: Iterable[float],
    error_budget: float,
    energy_function: Callable[[str, str], float],
) -> tuple[int, float, float]:
    """Choose the threshold for double-factorization that satisfies the error budget.

    Args:
        fci: A PySCF fcidump object.
        candidate_thresholds: The candidate thresholds.
        error_budget: The energy error budget (Ha).
        energy_function: A function that evaluates the energy of an fcidump object.

    Returns:
        The first threshold amongst the candidates that meets the energy error budget
            as well as the energy error ERI L2 error for this threshold.

    Raises:
        ValueError: If none of the candidate thresholds meets the energy error budget.
    """

    escf, ecor, etot = get_dmrg_energy(fci)
    eri_full = fci["H2"]

    exact_etot = etot

    best_threshold: Optional[int] = None
    for threshold in candidate_thresholds:
        eri_rr, LR, L, Lxi = df.factorize(eri_full, threshold)
        fci["H2"] = eri_rr

        try:
            energy = get_dmrg_energy(fci)
        except Exception as e:
            print(f"Threshold {threshold} failed: {e}")

        error = etot - exact_etot
        l2_norm_error_eri = np.linalg.norm(
            eri_rr - eri_full
        )  # eri reconstruction error

        print(
            f"Threshold: {threshold}, error: {error}, eri L2 error: {l2_norm_error_eri}"
        )

        if np.abs(error) < error_budget:
            return threshold, error, l2_norm_error_eri

    raise ValueError("None of the thresholds provided satisfies the error budget.")
