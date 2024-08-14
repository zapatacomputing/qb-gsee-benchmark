################################################################################
# © Copyright 2024 Zapata Computing Inc.
################################################################################


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


def get_df_qpe_circuit(
    fci: dict, target_accuracy: float, allowable_failure_rate: float
):
    instance = get_chemical_hamiltonian(fci)
    encoding = DoubleFactorized(instance=instance, prec=1e-10)
    circuit = QubitizedPhaseEstimation(
        encoding,
    )
    return circuit, 1000, 1e-3
