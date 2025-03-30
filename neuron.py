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

    

    def optimizer(self,neuron_output,true_value):
        """
        minimizing the out-true difference function

        Args:
            neuron_output (_type_): what we get
            true_value (_type_): and what we want
        """
        pass
        
# to be continued...
