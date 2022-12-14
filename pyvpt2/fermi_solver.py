import numpy as np

# interaction: {"left": {"state": [0], "nu": nu0}, "right": {"state": [1,1], "nu": nu11}, "phi": phi_011, "type": 1}
def fermi_solver(fermi_list):
    polyad_list = [] # list of constructed polyads
    for interaction in fermi_list:
        flag = False
        for polyad in polyad_list:
            if interaction["left"]["state"] in polyad.state_list:
                polyad.add(interaction)
                flag = True
            elif interaction["right"]["state"] in polyad.state_list:
                polyad.add(interaction)
                flag = True
            
        if flag == False:
            polyad_list.append(Polyad(interaction))

    state_list = {}
    for polyad in polyad_list:
        state_list.update(polyad.solve())

    return state_list # dict of states and their variationally corrected frequencies


class Polyad:
    def __init__(self, interaction):
        left = interaction.pop("left")
        right = interaction.pop("right")
        self.state_list = set([left["state"], right["state"]])
        self.nu_list = {left["state"]: left["nu"], right["state"]: right["nu"]}
        self.phi_list = {(left["state"], right["state"]): interaction}

    def add(self, interaction):
        """ add an interaction to the state list"""
        left = interaction.pop("left")
        right = interaction.pop("right")
        self.state_list.update([left["state"], right["state"]])
        self.nu_list.update({left["state"]: left["nu"]})
        self.nu_list.update({right["state"]: right["nu"]})
        self.phi_list.update({(left["state"], right["state"]): interaction})

    def build_hamiltonian(self):
        """ build effective hamiltonian from state list"""
        self.state_list_enum = {state: i for i, state in enumerate(self.state_list)} 
        dim = len(self.state_list_enum.keys())
        self.H = np.zeros((dim, dim))
        for state, i in self.state_list_enum.items():
            self.H[i, i] = self.nu_list[state]
        
        for states, interaction in self.phi_list.items():
            i = self.state_list_enum[states[0]]
            j = self.state_list_enum[states[1]]
            if interaction["type"] == 1:
                self.H[i, j] = 1/4 * interaction["phi"]
                self.H[j, i] = self.H[i, j]
            elif interaction["type"] == 2:
                self.H[i, j] = 1/(np.sqrt(2) * 2) * interaction["phi"]
                self.H[j, i] = self.H[i, j]
        
    def solve(self):
        """ solve hamiltonian"""
        self.build_hamiltonian()
        evals, evecs = np.linalg.eigh(self.H)
        inds = [np.argmax(vec) for vec in np.square(evecs.T)] # list of state indices of each eval
        eval_dict = dict(zip(inds,evals))

        freqs = { state: eval_dict[ind] for (state,ind) in self.state_list_enum.items() }
        return freqs
        
        

