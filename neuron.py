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
    def __init__(self, 
                norepinephrine_i: int = 0, norepinephrine_o: int = 0,
                acetylcholine_i: int = 0, acetylcholine_o: int = 0,
                dopamine_i: int = 0, dopamine_o: int = 0,
                gaba_i: int = 0, gaba_o: int = 0,
                glutamate_i: int = 0, glutamate_o: int = 0,
                glycine_i: int = 0, glycine_o: int = 0,
                histamine_i: int = 0, histamine_o: int = 0,
                opioid_i: int = 0, opioid_o: int = 0,
                serotonin_i: int = 0, serotonin_o: int = 0, history: int = 16
                ):
        self.Q=cp.ones(shape=(10,2)) # quality factors for input/output transmitter values
        """
        Initialize a Neuron instance with input/output connections for neurotransmitters.
        """
        
       

        self.neurotransmitters = {
            "norepinephrine": {"input": norepinephrine_i, "output": norepinephrine_o},
            "acetylcholine": {"input": acetylcholine_i, "output": acetylcholine_o},
            "dopamine": {"input": dopamine_i, "output": dopamine_o},
            "gaba": {"input": gaba_i, "output": gaba_o},
            "glutamate": {"input": glutamate_i, "output": glutamate_o},
            "glycine": {"input": glycine_i, "output": glycine_o},
            "histamine": {"input": histamine_i, "output": histamine_o},
            "opioid": {"input": opioid_i, "output": opioid_o},
            "serotonin": {"input": serotonin_i, "output": serotonin_o},
            
        }
        self.norepinephrine_i = cp.zeros(norepinephrine_i)
        self.norepinephrine_o = cp.zeros(norepinephrine_o)
        self.acetylcholine_i = cp.zeros(acetylcholine_i)
        self.acetylcholine_o = cp.zeros(acetylcholine_o)
        self.dopamine_i = cp.zeros(dopamine_i)
        self.dopamine_o = cp.zeros(dopamine_o)
        self.gaba_i = cp.zeros(gaba_i)
        self.gaba_o = cp.zeros(gaba_o)
        self.glutamate_i = cp.zeros(glutamate_i)
        self.glutamate_o = cp.zeros(glutamate_o)
        self.glycine_i = cp.zeros(glycine_i)
        self.glycine_o = cp.zeros(glycine_o)
        self.histamine_i = cp.zeros(histamine_i)
        self.histamine_o = cp.zeros(histamine_o)
        self.opioid_i = cp.zeros(opioid_i)
        self.opioid_o = cp.zeros(opioid_o)
        self.serotonin_i = cp.zeros(serotonin_i)
        self.serotonin_o = cp.zeros(serotonin_o)
        
        
        self.history_index = {nt: 0 for nt in self.neurotransmitters}
        # Validate inputs
        if any(cp.any(getattr(self, f"{nt}_i")) < 0 or cp.any(getattr(self, f"{nt}_o")) < 0 for nt in self.neurotransmitters):
            raise ValueError("Input/output counts for neurotransmitters must be non-negative integers.")


        # Logical constraints
        if acetylcholine_i +acetylcholine_o > 0 and dopamine_i + dopamine_o == 0:
            raise ValueError("If acetylcholine is used, dopamine must also be used.")
        if gaba_i + gaba_o > 0 and (
                dopamine_i + dopamine_o == 0 or
                glutamate_i +glutamate_o == 0 or
                acetylcholine_i +acetylcholine_o == 0):
            raise ValueError("GABA requires dopamine, acetylcholine, or glutamate.")
        if serotonin_i + serotonin_o > 0 and dopamine_i + dopamine_o == 0:
            raise ValueError("Serotonin requires dopamine.")
        if glycine_i + glycine_o > 0 and acetylcholine_i +acetylcholine_o == 0:
            raise ValueError("Glycine requires acetylcholine.")

        # Initialize arrays for neurotransmitter input/output
        for nt, counts in self.neurotransmitters.items():
            setattr(self, f"{nt}_i", cp.zeros(counts["input"]))  # Always create the input array
            setattr(self, f"{nt}_o", cp.zeros(counts["output"]))  # Always create the output array
        self.history_size = history  # Length of input memory
        self.input_history = {
            nt: cp.zeros((self.history_size, counts["input"]))
            for nt, counts in self.neurotransmitters.items()
        }


    def response(self):
        """
        Calculate responses for each neurotransmitter based on rules and set output arrays.
        """
        # Define stimuli and inhibitory rules for each neurotransmitter
        # note - only glutamate and/or dopamine will be considered "positive output", while others - negative or just auxiliary
        rules = {
            "norepinephrine": {
                "stimuli": ["acetylcholine", "dopamine", "glutamate", "serotonin", "histamine"],
                "inhibitory": ["gaba", "opioid", "glycine"]
            },
            "acetylcholine": {
                "stimuli": ["dopamine", "glutamate", "serotonin", "norepinephrine", "histamine"],
                "inhibitory": ["gaba", "glycine", "opioid"]
            },
            "dopamine": {
                "stimuli": ["glutamate", "norepinephrine", "histamine", "opioid"],
                "inhibitory": ["gaba", "acetylcholine", "serotonin", "glycine"]
            },
            "gaba": {
                "stimuli": ["dopamine", "glutamate", "glycine", "opioid"],
                "inhibitory": ["acetylcholine"]
            },
            "glutamate": {
                "stimuli": ["acetylcholine", "dopamine", "glycine"],
                "inhibitory": ["gaba", "opioid"]
            },
            "glycine": {
                "stimuli": ["gaba", "glutamate"],
                "inhibitory": ["dopamine", "acetylcholine"]
            },
            "histamine": {
                "stimuli": ["dopamine", "glutamate", "acetylcholine"],
                "inhibitory": ["gaba"]
            },
            "opioid": {
                "stimuli": ["dopamine", "gaba"],
                "inhibitory": ["glutamate", "serotonin"]
            },
            "serotonin": {
                "stimuli": ["acetylcholine", "dopamine", "norepinephrine"],
                "inhibitory": ["gaba"]
            },
            "norepinephrine": {
                "stimuli": ["acetylcholine", "dopamine", "glutamate", "serotonin", "histamine"],
                "inhibitory": ["gaba", "opioid", "glycine"]
            }
        }

        # Calculate responses dynamically
        for nt in self.neurotransmitters:
            stimuli = sum(cp.sum(getattr(self, f"{stim}_i")) for stim in rules[nt]["stimuli"])
            inhibitory = sum(cp.sum(getattr(self, f"{inh}_i")) for inh in rules[nt]["inhibitory"])
            response = stimuli - inhibitory

            # Assign response to output
            if getattr(self, f"{nt}_o") is not None:
                getattr(self, f"{nt}_o")[:] = response

    def update_input_history(self):
        """
        Store the latest input arrays for each neurotransmitter in fixed-length memory buffers.
        """
        for nt in self.neurotransmitters:
            input_array = getattr(self, f"{nt}_i")
            if input_array is not None and self.input_history.get(nt) is not None:
                idx = self.history_index[nt] % self.history_size
                self.input_history[nt][idx] = input_array
            self.history_index[nt] += 1    
    def optimizer(self, loss):
        def sigmoid(value):
            value = cp.asarray(value)
            # value = cp.clip(value, -500, 500)
            return 1.0 / (1.0 + cp.exp(-value)) + 1.0

        loss = cp.asarray(loss)
        if loss.size == 1:
            loss = cp.full((len(self.neurotransmitters),), loss)

        for i, (nt, data) in enumerate(self.neurotransmitters.items()):
            input_arr = cp.asarray(data["input"])
            output_arr = cp.asarray(data["output"])

            input_val = input_arr*self.Q[i,0] if input_arr.ndim > 0 else cp.asarray([float(input_arr)])*self.Q[i,0]
            output_val = output_arr*self.Q[i,1] if output_arr.ndim > 0 else cp.asarray([float(output_arr)])*self.Q[i,1]
            # print(output_val)

            # Apply per-receptor loss delta
            delta_in = (input_val - loss[i]) ** 2
            delta_out = (output_val - loss[i]) ** 2

            # Apply sigmoid scaled inversely by Q (less certain = steeper adaptation)
            q_in = sigmoid(delta_in )
            q_out = sigmoid(delta_out)

            self.Q[i, 0] = float(cp.mean(q_out))
            self.Q[i, 1] = float(cp.mean(q_in))
            # print(f"Optimizer [{nt}]: Δin={cp.mean(delta_in):.4f}, Δout={cp.mean(delta_out):.4f}, Q_in={self.Q[i,0]:.4f}, Q_out={self.Q[i,1]:.4f}")



            
    def backpropagate(self, loss):
        self.callback = loss
        learning_rate = 0.001
        loss = cp.asarray(loss)

        # Broadcast scalar loss to match neurotransmitter count
        if loss.size == 1:
            loss = cp.full((len(self.neurotransmitters),), loss)

        for i, (nt, data) in enumerate(self.neurotransmitters.items()):
            output_val = cp.asarray(data["output"])
            if not hasattr(output_val, "__len__"):
                output_val = cp.asarray([float(output_val)])

            error = loss[i]

            # if self.input_history.get(nt) is not None:
            input_history = self.input_history[nt]
            weighted_input = input_history * self.Q[i, 0]
            averaged_input = cp.mean(weighted_input)
            # else:
                # averaged_input = cp.asarray(data["input"])

            weighted_output = output_val * self.Q[i, 1]
            
            dQ_input = 2 * error * averaged_input
            dQ_output = 2 * error * weighted_output

            self.Q[i, 0] += learning_rate * dQ_input
            self.Q[i, 1] += learning_rate * dQ_output

        # print(self.Q)
            
            
        
        self.update_input_history()
        self.optimizer(loss)
        self.response()

        





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


      
        
        # 
        #                   ======================================
        #                              Sample usage trial
        #                   ======================================
        # 
'''
# assume we have only 1 sensor
sensor_data=1.0
loss=1.1
#   declaration of neurons - sample usage
module1=Neuron(acetylcholine_i=1,dopamine_o=2,acetylcholine_o=2,opioid_o=1,norepinephrine_o=1)
module2=Neuron(opioid_i=1,dopamine_i=1,acetylcholine_i=1,dopamine_o=1,opioid_o=1)
module3=Neuron(norepinephrine_i=1,dopamine_i=1,dopamine_o=1,acetylcholine_i=1,acetylcholine_o=1,opioid_o=1)
module_final=Neuron(opioid_i=2,dopamine_i=2,acetylcholine_i=1,dopamine_o=1)
# Connections now, yeah - i'll make it easier, maybe someday... all inside loop

print(sensor_data)

def expa(val): # dummy response for network action
    return cp.exp(-val/2)

for i in range(1000):
    module1.acetylcholine_i[0] = sensor_data
    module2.dopamine_i[0], module2.opioid_i[0], module2.acetylcholine_i[0], \
    module3.norepinephrine_i[0], module3.dopamine_i[0], module3.acetylcholine_i[0] = \
        module1.dopamine_o[0], module1.opioid_o[0], module1.acetylcholine_o[0], \
        module1.norepinephrine_o[0], module1.dopamine_o[1], module1.acetylcholine_o[1]

    module_final.opioid_i[0], module_final.opioid_i[1], \
    module_final.dopamine_i[0], module_final.dopamine_i[1], module_final.acetylcholine_i[0] = \
        module2.opioid_o[0], module3.opioid_o[0], module3.dopamine_o[0], \
        module2.dopamine_o[0], module3.acetylcholine_o[0]

    out = module_final.dopamine_o[0]
    sensor_data = 0.5 * out + 0.5 * loss

    # Backpropagation for each module to minimize the error
    module1.backpropagate(loss)
    module2.backpropagate(loss)
    module3.backpropagate(loss)
    module_final.backpropagate(loss)

    # module1.threshold()
    # module2.threshold()
    # module3.threshold()
    # module_final.threshold()

    print(i,"\t", sensor_data, "\t",loss, "\t",out)
    if i % 20 == 0:
        loss += 10
'''
