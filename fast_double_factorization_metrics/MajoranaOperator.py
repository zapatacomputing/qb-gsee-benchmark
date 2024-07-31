import copy
import numpy as np

from openfermion.ops import QubitOperator

from GeneralOperator import GeneralOperator
import utils.majorana_utils as mu
import utils.jw_utils as jw



class MajoranaOperator(GeneralOperator):

    def __init__(self, data, num_modes, normal_ordered=True, skip_initial_ordering=False):
        if (normal_ordered and (not skip_initial_ordering)):
            self.data = mu._normal_order_majorana_data(data, num_modes)
        else:
            self.data = data

        self.num_modes = num_modes
        self.normal_ordered = normal_ordered

    def __str__(self):
        return mu._majorana_data_to_str(self.data)
    
    def normal_order(self):
        # Make normal ordered, in place
        self.data = mu._normal_order_majorana_data(self.data, num_modes=self.num_modes)
        self.normal_ordered = True
    
    @classmethod
    def _multiply_elements(cls, element1, element2, **kwargs):
        return mu._multiply_maj_integers(element1, element2, kwargs['num_modes'], normal_order=kwargs['normal_ordered'])

    def dagger(self):
        # majoranas are Hermitian, but dagger reverses order

        if self.normal_ordered:
            # if we can, use the normal order method
            return self._normal_order_dagger()
        
        ret = copy.deepcopy(self)
        conj_data = {key[::-1]: np.conj(value) for key, value in self.data.items()}
        if self.normal_ordered:
            conj_data = mu._normal_order_majorana_data(conj_data, self.num_modes)
        ret.data = conj_data
        return ret

    def _normal_order_dagger(self):
        # Simple way to get conjugate in normal order format
        conj_data = {}
        for key, val in self.data.items():
            m = len(key) % 4
            if m == 0 or m == 1:
                conj_data[key] = val
            elif m == 2 or m == 3:
                conj_data[key] = -1 * val

        ret = copy.deepcopy(self)
        ret.data = conj_data
        return ret

    def trace(self):
        if self.normal_ordered:
            return self.data[()]
        else:
            raise UserWarning("Trace not implemented for un-ordered majorana operators.")

    def trace_norm(self, *args, **kwargs):
        return super().trace_norm(*args, **kwargs)
    
    def jordan_wigner_transform(self, use_openfermion=True):
        if not self.normal_ordered:
            raise NotImplementedError
        
        majorana_data = self.data

        pauli_dict = {}
        for majorana_ints, value in majorana_data.items():
                
            pauli_letters, coeff = jw.get_jw_paulis(majorana_ints, value)

            pauli_dict[tuple(pauli_letters)] = coeff

        if use_openfermion:
            H_op = QubitOperator(term=None)
            for pauli_letters, coeff in pauli_dict.items():
                H_op += QubitOperator(pauli_letters, coeff)

            return H_op

        return pauli_dict