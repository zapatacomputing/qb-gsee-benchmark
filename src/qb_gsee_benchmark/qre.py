################################################################################
# © Copyright 2024 Zapata Computing Inc.
################################################################################
import numpy as np
from openfermion import MolecularData
from openfermion.resource_estimates.molecule import cas_to_pyscf
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


def get_chemical_hamiltonian(fci) -> ChemicalHamiltonian:
    num_alpha = (fci["NELEC"] + fci["MS2"]) // 2
    num_beta = num_alpha - fci["MS2"]
    eri_full = ao2mo.restore("s1", fci["H2"], fci["H1"].shape[0])

    active_space_molecule, active_space_meanfield_object = cas_to_pyscf(
        fci["H1"], eri_full, fci["ECORE"], num_alpha, num_beta
    )

    molecular_data = _get_molecular_data(
        active_space_molecule, active_space_meanfield_object
    )

    interaction_op = molecular_data.get_molecular_hamiltonian()
    return ChemicalHamiltonian(mol_ham=interaction_op, mol_name="H2")


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
    fci: dict, square_overlap: float, error_tolerance: float, failure_tolerance: float
):
    """Get the QPE circuit for a given PySCF FCI object.

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
    phase_estimation_failure_tolerance = failure_tolerance * 0.8
    state_projection_failure_tolerance = failure_tolerance * 0.1
    hardware_failure_tolerance = failure_tolerance * 0.1

    phase_estimation_error_tolerance = error_tolerance * 0.9
    walk_operator_error_tolerance = error_tolerance * 0.1

    num_shots = get_num_shots(square_overlap, state_projection_failure_tolerance)

    instance = get_chemical_hamiltonian(fci)
    encoding = DoubleFactorized(
        instance=instance, energy_error=walk_operator_error_tolerance * 10, prec=1e-10
    )
    circuit = QubitizedPhaseEstimation(
        encoding,
    )

    hardware_failure_tolerance_per_shot = get_hardware_failure_tolerance_per_shot(
        failure_tolerance=hardware_failure_tolerance, num_shots=num_shots
    )

    return circuit, num_shots, hardware_failure_tolerance_per_shot
