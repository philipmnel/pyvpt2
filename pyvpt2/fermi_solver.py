import numpy as np
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class State:
    """
    Dataclass to store a state

    Attributes
    ----------
    state : tuple of int
        state indices
    nu : float
        state frequency
    """
    state: tuple
    nu: float

@dataclass
class Interaction:
    """
    Dataclass to store an interaction

    Attributes
    ----------
    left : State
        left state and its frequency
    right : State
        right state and its frequency
    phi : float
        interaction strength
    ftype : int
        Fermi type (1 or 2)
    """
    left: State
    right: State
    phi: float
    ftype: int

def fermi_solver(interaction_list: List) -> Dict:
    """ 
    Main loop to correct Fermi resonances. Constructs polyads from the interaction list 
    and then solves the effective Hamiltonian for each.

    Parameters
    ----------
    interaction_list : List
        list of Interaction objects

    Returns
    -------
    Dict
        dictionary of states and their variationally corrected frequencies
    """
    polyad_list = [] # list of constructed polyads
    for interaction in interaction_list:
        flag = False
        for polyad in polyad_list:
            if interaction.left.state in polyad.state_list:
                polyad.add(interaction)
                flag = True
            elif interaction.right.state in polyad.state_list:
                polyad.add(interaction)
                flag = True
            
        if flag == False:
            polyad_list.append(Polyad(interaction))

    state_list = {}
    for polyad in polyad_list:
        state_list.update(polyad.solve())

    return state_list 


class Polyad:
    """
    Class to construct an interaction polyad and solve the effective Hamiltonian
    
    Attributes
    ----------
    state_list : set
        set of states in the polyad
    nu_list : Dict
        dictionary of states and their frequencies
    phi_list : Dict
        dictionary of interacting states and their phi value (off-diagonal elements of H)
    state_list_enum : Dict
        dictionary of states and their enumeration
    H : np.ndarray
        effective Hamiltonian
    """
    def __init__(self, interaction: Interaction):
        """ 
        Initialize polyad with first interaction
        
        Parameters
        ----------
        interaction : Interaction
        """

        left = interaction.left
        right = interaction.right
        self.state_list = set([left.state, right.state])
        self.nu_list = {left.state: left.nu, right.state: right.nu}
        self.phi_list = {(left.state, right.state): (interaction.phi, interaction.ftype)}

    def add(self, interaction: Interaction):
        """ 
        Add an interaction to the state list
        
        Parameters
        ----------
        interaction : Interaction
        """
        left = interaction.left
        right = interaction.right
        self.state_list.update([left.state, right.state])
        self.nu_list.update({left.state: left.nu})
        self.nu_list.update({right.state: right.nu})
        self.phi_list.update({(left.state, right.state): (interaction.phi, interaction.ftype)})

    def build_hamiltonian(self):
        """ 
        Build effective hamiltonian from state list
        """
        self.state_list_enum = {state: i for i, state in enumerate(self.state_list)} 
        dim = len(self.state_list_enum.keys())
        self.H = np.zeros((dim, dim))
        for state, i in self.state_list_enum.items():
            self.H[i, i] = self.nu_list[state]
        
        for states, interaction in self.phi_list.items():
            i = self.state_list_enum[states[0]]
            j = self.state_list_enum[states[1]]
            if interaction[1] == 1:
                self.H[i, j] = 1/4 * interaction[0]
                self.H[j, i] = self.H[i, j]
            elif interaction[1] == 2:
                self.H[i, j] = 1/(np.sqrt(2) * 2) * interaction[0]
                self.H[j, i] = self.H[i, j]
        
    def solve(self) -> Dict:
        """
        Solve hamiltonian
        
        Returns
        -------
        Dict
            dictionary of states and their corrected frequencies
        """
        self.build_hamiltonian()
        evals, evecs = np.linalg.eigh(self.H)
        inds = [np.argmax(vec) for vec in np.square(evecs.T)] # list of state indices of each eval
        eval_dict = dict(zip(inds,evals))

        freqs = { state: eval_dict[ind] for (state,ind) in self.state_list_enum.items() }
        return freqs
        