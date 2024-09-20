################################################################################
# © Copyright 2024 Zapata Computing Inc.
################################################################################
from typing import Iterable, Optional

import numpy as np
from openfermion import MolecularData
from openfermion.resource_estimates import df
from openfermion.resource_estimates.molecule import cas_to_pyscf, factorized_ccsd_t
from openfermionpyscf import PyscfMolecularData
from openfermionpyscf._run_pyscf import compute_integrals
from pyLIQTR.BlockEncodings.DoubleFactorized import DoubleFactorized
from pyLIQTR.ProblemInstances.ChemicalHamiltonian import ChemicalHamiltonian
from pyLIQTR.qubitization.phase_estimation import QubitizedPhaseEstimation
from pyscf import ao2mo, gto, scf


def _get_molecular_data(
    molecule: scf.hf.SCF,
    mean_field_object: gto.Mole,
) -> PyscfMolecularData:
    """Given a PySCF meanfield object and molecule, return a PyscfMolecularData
    object.

    Returns:
        A PyscfMolecularData object corresponding to the meanfield object and
            molecule.

    Raises:
        SCFConvergenceError: If the SCF calculation does not converge.
    """
    molecular_data = MolecularData(
        geometry=[
            ("H", (0, 0, 0)),
        ],  # Dummy geometry needed for naming purposes
        basis=molecule.basis,
        multiplicity=molecule.spin + 1,
        charge=molecule.charge,
    )

    molecular_data.n_orbitals = int(molecule.nao)
    molecular_data.n_qubits = 2 * molecular_data.n_orbitals
    molecular_data.nuclear_repulsion = float(molecule.energy_nuc())

    molecular_data.hf_energy = float(mean_field_object.e_tot)

    molecular_data._pyscf_data = {  # type: ignore
        "mol": molecule,
        "scf": mean_field_object,
    }

    molecular_data.canonical_orbitals = mean_field_object.mo_coeff.astype(float)
    molecular_data.orbital_energies = mean_field_object.mo_energy.astype(float)

    one_body_integrals, two_body_integrals = compute_integrals(
        mean_field_object._eri, mean_field_object
    )
    molecular_data.one_body_integrals = one_body_integrals
    molecular_data.two_body_integrals = two_body_integrals
    molecular_data.overlap_integrals = mean_field_object.get_ovlp()

    pyscf_molecular_data = PyscfMolecularData.__new__(PyscfMolecularData)
    pyscf_molecular_data.__dict__.update(molecule.__dict__)

    return molecular_data


def choose_double_factorization_threshold(
    meanfield_object: scf.hf.SCF,
    eri_full: np.array,
    candidate_thresholds: Iterable[float],
    error_budget: float,
    use_kernel=True,
    use_triples=True,
) -> tuple[int, float, float]:
    """Choose the threshold for double-factorization that satisfies the error budget.

    Note that this function requires that the _eri attribute of the meanfield object be
        shaped as a four-index tensor.

    Args:
        meanfield_object: The mean-field object.
        candidate_thresholds: The candidate thresholds.
        error_budget: The energy error budget (Ha).
        use_kernel: Whether to use the kernel.
        use_triples: Whether to use the triples.

    Returns:
        The first threshold amongst the candidates that meets the energy error budget
            as well as the energy error ERI L2 error for this threshold.

    Raises:
        ValueError: If none of the candidate thresholds meets the energy error budget.
    """

    escf, ecor, etot = factorized_ccsd_t(
        meanfield_object, eri_rr=None, use_kernel=use_kernel, no_triples=not use_triples
    )

    exact_etot = etot

    best_threshold: Optional[int] = None
    for threshold in candidate_thresholds:
        eri_rr, LR, L, Lxi = df.factorize(eri_full, threshold)

        try:
            escf, ecor, etot = factorized_ccsd_t(
                meanfield_object,
                eri_rr,
                use_kernel=use_kernel,
                no_triples=not use_triples,
            )
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


def get_num_shots(square_overlap: float, failure_tolerance: float):
    """Get the number of QPE shots required to achieve a given eigenstate projection
     failure tolerance.

    Corresponds to Eq. 15 in arXiv:2406.06335v1.

    Args:
        square_overlap: The square overlap between the target state and the initial
            state.
        failure_tolerance: The allowable probability that none of the shots projects on
            to the target state.
    """
    assert (
        square_overlap >= 0 and square_overlap <= 1
    ), "Square overlap must be in [0, 1]"
    return np.ceil(np.log(failure_tolerance) / np.log(1 - square_overlap))


def get_hardware_failure_tolerance_per_shot(failure_tolerance: float, num_shots: int):
    """Get the allowable hardware failure rate per shot.

    Corresponds to Eq. 18 in arXiv:2406.06335v1.

    Args:
        num_shots: The number of shots.
        failure_tolerance: The allowable probability that none of the shots experiences
            an (uncorrected) hardware error.
    """
    return 1 - (1 - failure_tolerance) ** (1 / num_shots)


def get_df_qpe_circuit(
    fci: dict,
    square_overlap: float,
    error_tolerance: float,
    failure_tolerance: float,
    candidate_thresholds: Optional[Iterable[float]] = None,
):
    """Get the QPE circuit for a given PySCF FCI object.

    This uses an algorithm performance model based on that described in
    arXiv:2406.06335v1 to account for the number of shots required to achieve the target
    success probability.

    Args:
        fci: The FCI object, i.e. what you get from loading a pyscf fcidump.
        square_overlap: The square overlap between the target state and the initial
            state.
        error_tolerance: The desired error in the energy estimation.
        failure_tolerance: The allowable probability that the absolute energy error
            exceeds the error tolerance.

    Returns:
        A tuple containing the QPE circuit, the number of shots, and the allowable
            hardware failure rate per shot.
    """
    if candidate_thresholds is None:
        candidate_thresholds = np.power(10.0, np.arange(-1, -6, -0.5))

    # Allowed probability that spectral leakage causes the estimated energy to deviate
    # from the true eigenvalue of the encoded Hamiltonian by more than
    # phase_estimation_error_tolerance. Corresponds to \overline{\delta_{QPE}} in
    # the paper.
    phase_estimation_failure_tolerance = failure_tolerance * 0.8

    # Allowed probability that none of the shots projects into the ground state.
    # Corresponds to \overline{\p_GS} in the paper.
    state_projection_failure_tolerance = failure_tolerance * 0.1

    # Allowed probability that one or more shots experiences an (uncorrected) hardware
    # error. Corresponds to \overline{delta_HW} in the paper.
    hardware_failure_tolerance = failure_tolerance * 0.1

    # Allowable error in the estimated energy due to spectral leakage. Corresponds to
    # \overline{\epsilon_{SL}} in the paper.
    spectral_leakage_error_tolerance = error_tolerance * 0.9

    # Allowable error in the energy due to the approximate nature of the block encoding.
    # Corresponds to \overline{\epsilon_{BE}} in the paper.
    block_encoding_error_tolerance = error_tolerance * 0.1

    # The allowed standard deviation of the spectral leakage for a single shot in order
    # to achieve the desired phase estimate failure rate based on the Chebyshev
    # inequality. See Eq. 8 in the paper.
    sigma = (
        np.sqrt(phase_estimation_failure_tolerance) * spectral_leakage_error_tolerance
    )

    num_shots = get_num_shots(square_overlap, state_projection_failure_tolerance)

    num_alpha = (fci["NELEC"] + fci["MS2"]) // 2
    num_beta = num_alpha - fci["MS2"]
    eri_full = ao2mo.restore("s1", fci["H2"], fci["H1"].shape[0])

    active_space_molecule, active_space_meanfield_object = cas_to_pyscf(
        fci["H1"], eri_full, fci["ECORE"], num_alpha, num_beta
    )

    (
        df_threshold,
        df_energy_error,
        df_eri_l2_error,
    ) = choose_double_factorization_threshold(
        meanfield_object=active_space_meanfield_object,
        eri_full=eri_full,
        candidate_thresholds=candidate_thresholds,
        error_budget=block_encoding_error_tolerance / 3,
    )

    molecular_data = _get_molecular_data(
        active_space_molecule, active_space_meanfield_object
    )

    interaction_op = molecular_data.get_molecular_hamiltonian()
    instance = ChemicalHamiltonian(mol_ham=interaction_op, mol_name="H2")

    # Note that the factor of ten arises because because PyLIQTR will multiply by 0.1 to
    # determine the allowed walk operator error. The factor of 2/3 comes from the fact
    # that we allocate 1/3 of the block encoding error to truncation.
    encoding = DoubleFactorized(
        instance,
        energy_error=block_encoding_error_tolerance * 10 * 2 / 3,
        df_error_threshold=df_threshold,
        sf_error_threshold=1e-12,
    )

    circuit = QubitizedPhaseEstimation(encoding, eps=sigma)

    hardware_failure_tolerance_per_shot = get_hardware_failure_tolerance_per_shot(
        failure_tolerance=hardware_failure_tolerance, num_shots=num_shots
    )

    return circuit, num_shots, hardware_failure_tolerance_per_shot
