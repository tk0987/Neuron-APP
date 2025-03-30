"""
NEURON PROJECT 
first sketch

basic neuron class (main)

new and own approach to ML

author: T. Kowalski 
"""

import sys
try:
    import cupy as cp
except ImportError:
    import numpy as cp


class Neuron:
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
                adrenaline_i: int = 0, adrenaline_o: int = 0,
                acetylcholine_i: int = 0, acetylcholine_o: int = 0,
                dopamine_i: int = 0, dopamine_o: int = 0,
                gaba_i: int = 0, gaba_o: int = 0,
                glutamate_i: int = 0, glutamate_o: int = 0,
                glycine_i: int = 0, glycine_o: int = 0,
                histamine_i: int = 0, histamine_o: int = 0,
                opioid_i: int = 0, opioid_o: int = 0,
                serotonin_i: int = 0, serotonin_o: int = 0,
                norepinephrine_i: int = 0, norepinephrine_o: int = 0):
        self.Q=cp.ones(shape=(10,2)) # quality factors for input/output transmitter values
        """
        Initialize a Neuron instance with input/output connections for neurotransmitters.
        """
        self.neurotransmitters = {
            "adrenaline": {"input": adrenaline_i, "output": adrenaline_o},
            "acetylcholine": {"input": acetylcholine_i, "output": acetylcholine_o},
            "dopamine": {"input": dopamine_i, "output": dopamine_o},
            "gaba": {"input": gaba_i, "output": gaba_o},
            "glutamate": {"input": glutamate_i, "output": glutamate_o},
            "glycine": {"input": glycine_i, "output": glycine_o},
            "histamine": {"input": histamine_i, "output": histamine_o},
            "opioid": {"input": opioid_i, "output": opioid_o},
            "serotonin": {"input": serotonin_i, "output": serotonin_o},
            "norepinephrine": {"input": norepinephrine_i, "output": norepinephrine_o},
        }
    

        # Validate inputs
        if any(getattr(self, f"{nt}_i") < 0 or getattr(self, f"{nt}_o") < 0 for nt in self.neurotransmitters):
            raise ValueError("Input/output counts for neurotransmitters must be non-negative integers.")

        # Logical constraints
        if self.acetylcholine_i + self.acetylcholine_o > 0 and self.dopamine_i + self.dopamine_o == 0:
            raise ValueError("If acetylcholine is used, dopamine must also be used.")
        if self.gaba_i + self.gaba_o > 0 and (
                self.dopamine_i + self.dopamine_o == 0 or
                self.glutamate_i + self.glutamate_o == 0 or
                self.acetylcholine_i + self.acetylcholine_o == 0):
            raise ValueError("GABA requires dopamine, acetylcholine, or glutamate.")
        if self.serotonin_i + self.serotonin_o > 0 and self.dopamine_i + self.dopamine_o == 0:
            raise ValueError("Serotonin requires dopamine.")
        if self.glycine_i + self.glycine_o > 0 and self.acetylcholine_i + self.acetylcholine_o == 0:
            raise ValueError("Glycine requires acetylcholine.")

        # Initialize arrays for neurotransmitter input/output
        for nt in self.neurotransmitters:
            if getattr(self, f"{nt}_i") > 0:
                setattr(self, f"{nt}_i", cp.zeros(getattr(self, f"{nt}_i")))
            if getattr(self, f"{nt}_o") > 0:
                setattr(self, f"{nt}_o", cp.zeros(getattr(self, f"{nt}_o")))

    def response(self):
        """
        Calculate responses for each neurotransmitter based on rules and set output arrays.
        """
        # Define stimuli and inhibitory rules for each neurotransmitter
        # note - only glutamate and/or dopamine will be considered "positive output", while others - negative or just auxiliary
        rules = {
            "adrenaline": {
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
                "stimuli": ["dopamine", "glutamate", "glycine", "acetylcholine", "opioid"],
                "inhibitory": ["dopamine", "glutamate", "acetylcholine"]
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
            stimuli = sum(cp.sum(getattr(self, f"{stim}_i", 0)) for stim in rules[nt]["stimuli"])
            inhibitory = sum(cp.sum(getattr(self, f"{inh}_i", 0)) for inh in rules[nt]["inhibitory"])
            response = stimuli - inhibitory

            # Assign response to output
            if getattr(self, f"{nt}_o", None) is not None:
                getattr(self, f"{nt}_o")[:] = response

    
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
            self.Q[i,:]=sigmoid((data["output"]-data["input"])-callback_value) # this approach means that neuron should amplify data...
            data["input"]*=self.Q[i,0]
            data["output"]*=self.Q[i,0]
            i+=1
    def threshold(self):
        self.thresholds=cp.ones_like(self.Q)
        i=0
        for nt, data in self.neurotransmitters.items():
            self.thresholds[i,0]*=data["input"]
            self.thresholds[i,1]*=data["output"]
            i+=1
        self.optimizer()
        i=0
        for nt, data in self.neurotransmitters.items():
            if data["input"]>self.thresholds[i,0]:
                self.response()
            else:
                data["output"]=0.0
            i+=1
            
        
        
    def IOstream(self):
        pass     
        # to be continued
