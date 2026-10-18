from dataclasses import dataclass
from typing import Dict, List

import numpy as np


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
        # every polyad that already holds one of the two states; an interaction that bridges several merges them into one
        hits = [polyad for polyad in polyad_list
                if interaction.left.state in polyad.state_list or interaction.right.state in polyad.state_list]
        if not hits:
            polyad_list.append(Polyad(interaction))
            continue
        target = hits[0]
        for other in hits[1:]:
            target.merge(other)
            polyad_list.remove(other)
        target.add(interaction)

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

    def merge(self, other: "Polyad"):
        """
        Absorb the states and interactions of another polyad

        Parameters
        ----------
        other : Polyad
        """
        self.state_list.update(other.state_list)
        self.nu_list.update(other.nu_list)
        self.phi_list.update(other.phi_list)

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
        # one eigenvalue per state: repeatedly take the largest remaining squared coefficient (state row, eigenvector column) and
        # retire both; taking each eigenvector's own largest component instead can give two eigenvectors the same state
        weight = np.square(evecs)
        eval_of_state = {}
        for _ in range(len(evals)):
            ind, vec = np.unravel_index(np.argmax(weight), weight.shape)
            eval_of_state[ind] = evals[vec]
            weight[ind, :] = -1.0
            weight[:, vec] = -1.0

        freqs = { state: eval_of_state[ind] for (state,ind) in self.state_list_enum.items() }
        return freqs
