"""
NEURON PROJECT 
first sketch

basic neuron class (main)

new and own approach to ML

author: T. Kowalski 
"""

# import sys
# try:
#     import cupy as cp
# except ImportError:
import numpy as cp
import pickle

RECEPTOR_RULES = {
    "AMPA": {"stimuli": ["NMDA", "D1", "5-HT2A", "alpha1A"], "inhibitory": ["GABA_A", "GABA_B", "CB1", "D2"]},
    "NMDA": {"stimuli": ["AMPA", "D1", "5-HT2A", "H2"], "inhibitory": ["GABA_A", "5-HT1A", "CB1", "sigma1", "GlyR alpha1"]},
    "Kainate": {"stimuli": ["AMPA", "mGluR1"], "inhibitory": ["GABA_A"]},
    "mGluR1": {"stimuli": ["NMDA", "5-HT2A"], "inhibitory": ["GABA_B"]},
    "mGluR2": {"inhibitory": ["Glutamate release"]},
    "GABA_A": {"stimuli": ["5-HT1A", "D2", "Mu (μ)"], "inhibitory": ["AMPA", "NMDA", "H1", "NK1"]},
    "GABA_B": {"stimuli": ["D2", "CB1"], "inhibitory": ["AMPA", "mGluR1", "H2"]},
    "D1": {"stimuli": ["NMDA", "AMPA", "5-HT2A", "H1"], "inhibitory": ["GABA_A", "D2"]},
    "D2": {"stimuli": ["GABA_B", "5-HT1A", "H2"], "inhibitory": ["D1", "AMPA"]},
    "5-HT1A": {"stimuli": ["GABA_A", "D2"], "inhibitory": ["NMDA", "5-HT2A"]},
    "5-HT2A": {"stimuli": ["D1", "NMDA", "mGluR1"], "inhibitory": ["GABA_A", "5-HT1A"]},
    "5-HT3": {"stimuli": ["GABAergic interneurons"]},
    "H1": {"stimuli": ["D1", "5-HT2C"], "inhibitory": ["GABA_A", "MT1"]},
    "H2": {"stimuli": ["D2", "NMDA"], "inhibitory": ["GABA_B"]},
    "H3": {"inhibitory": ["H1", "H2"]},
    "CB1": {"stimuli": ["GABA_B"], "inhibitory": ["NMDA", "AMPA", "Glutamate release"]},
    "Mu (μ)": {"stimuli": ["GABA_A"], "inhibitory": ["NMDA", "5-HT2A"]},
    "GlyR alpha1": {"inhibitory": ["NMDA"]},
    "alpha1A": {"stimuli": ["AMPA"]},
    "alpha2A": {"inhibitory": ["D1", "5-HT2A"]},
    "sigma1": {"inhibitory": ["NMDA"]},
    "MT1": {"inhibitory": ["H1"]},
    "OX1": {"stimuli": ["Histamine release"]},
    "NK1": {"inhibitory": ["GABA_A"]}
}
class Neuron():
    '''
    how it should work:
    similiar to dense keras unit, but capable of transferring different data at time - for example
    stimulatory dopamine and glutamate at once, and inhibitory/mediatory others
    
    note, that in __init__ we have max number of input/output connections, and not they values
    
    in nature we have only positive concentrations of transmitters, but here they are described by floats. 
    neuron will contact with another one or effector, if float describing neurotransmitter value will be above 
    threshold value. 
    optimization will be applied as a constant 'Q' - float - and total response will be multiplied with Q. same will apply for incoming signals
    
    receptor subtypes are omitted for simplicity.
    '''
class Neuron:
    def __init__(self, receptor_config=None, callback_val=None, history=16):
        """
        receptor_config: dict of receptor_name -> {"input": int, "output": int}
        """
        self.receptors = {}
        self.Q = cp.ones((len(RECEPTOR_RULES), 2))  # Quality factors
        self.callback = callback_val
        self.history_size = history

        # Initialize receptor I/O arrays
        for idx, receptor in enumerate(RECEPTOR_RULES):
            input_count = receptor_config.get(receptor, {}).get("input", 0) if receptor_config else 0
            output_count = receptor_config.get(receptor, {}).get("output", 0) if receptor_config else 0
            self.receptors[receptor] = {
                "input": cp.zeros(input_count),
                "output": cp.zeros(output_count),
                "history": cp.zeros((history, input_count)) if input_count > 0 else None
            }

    def response(self):
        """
        Calculate receptor responses based on dynamic rules.
        """
        for idx, (receptor, config) in enumerate(self.receptors.items()):
            rules = RECEPTOR_RULES.get(receptor, {})
            stimuli_sum = 0
            inhibitory_sum = 0

            for stim in rules.get("stimuli", []):
                if stim in self.receptors:
                    stimuli_sum += cp.sum(self.receptors[stim]["input"])

            for inh in rules.get("inhibitory", []):
                if inh in self.receptors:
                    inhibitory_sum += cp.sum(self.receptors[inh]["input"])

            response = (stimuli_sum - inhibitory_sum) * self.Q[idx, 0]

            if config["output"].size > 0:
                config["output"][:] = response * self.Q[idx, 1]

    def update_input_history(self):
        """
        Store the latest input arrays for each neurotransmitter in fixed-length memory buffers.
        """
        for nt in self.neurotransmitters:
            input_array = getattr(self, f"{nt}_i", None)
            if input_array is not None and self.input_history.get(nt) is not None:
                idx = self.history_index[nt] % self.history_size
                self.input_history[nt][idx] = input_array
            self.history_index[nt] += 1    
    def optimizer(self,callback_value):
        
        """
        minimizing the out-true difference function.
        
        

        Args:
            neuron_input (_type_): what we feed
        
            neuron_output (_type_): what we get
            
            callback_value (_type_): and what we want
            
        """
        def sigmoid(value):
            return 1.0/(1.0+cp.exp(-1.0*value))+1.0
        
        i=0
        for nt, data in self.neurotransmitters.items():
            self.Q[i,:]=sigmoid(((data["output"]-data["input"])**2-callback_value**2)**3) # this approach means that neuron should amplify data...
            data["input"]*=self.Q[i,0]
            data["output"]*=self.Q[i,1]
            i+=1
    def threshold(self):
        self.update_input_history()

        self.thresholds=cp.ones_like(self.Q)
        i=0
        for nt, data in self.neurotransmitters.items():
            self.thresholds[i,0]*=data["input"]
            self.thresholds[i,1]*=data["output"]
            i+=1
        self.optimizer(self.callback)
        i=0
        for nt, data in self.neurotransmitters.items():
            if data["input"]>self.thresholds[i,0]:
                self.response()
            else:
                data["output"]=0.0
            i+=1
            
    def backpropagate(self, target_value):
        """
        Perform backpropagation using the average of recent input history.
        """
        learning_rate = 0.01

        for i, (nt, data) in enumerate(self.neurotransmitters.items()):
            output_error = data["output"] - target_value

            # Use averaged input history
            if self.input_history.get(nt) is not None:
                averaged_input = cp.mean(self.input_history[nt], axis=0)
            else:
                averaged_input = cp.zeros_like(data["input"])

            dQ_input = 2 * output_error * averaged_input
            dQ_output = 2 * output_error * data["output"]

            self.Q[i, 0] -= learning_rate * dQ_input
            self.Q[i, 1] -= learning_rate * dQ_output

            # Optional: clip extremes
            self.Q[i, :] = cp.clip(self.Q[i, :], 0.1, 10.0)
    def save(self,path):

        # Save your neuron network to a binary file
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load():

        with open("neuron_network.pkl", "rb") as f:
            neuron = pickle.load(f)
        return neuron
    @staticmethod
    def connect_neurons(source, target, neurotransmitter):
        """
        Connects the output values of 'source' neuron to the input of 'target' neuron via a specific neurotransmitter.
        """
        source_output = getattr(source, f"{neurotransmitter}_o")
        target_input = getattr(target, f"{neurotransmitter}_i")

        # Clip or broadcast if sizes differ
        min_len = min(len(source_output), len(target_input))
        target_input[:min_len] += source_output[:min_len]

'''        
        
        # 
        #                   ======================================
        #                              Sample usage 
        #                   ======================================
        # 

config = {
    "AMPA": {"input": 3, "output": 2},
    "NMDA": {"input": 2, "output": 1},
    "D1": {"input": 1, "output": 1},
    # Add more as needed...
}

neuron = Neuron(receptor_config=config)
neuron.receptors["NMDA"]["input"][:] = cp.array([1.0, 0.5])
neuron.response()
print(neuron.receptors["AMPA"]["output"])

'''
