"""
NEURON PROJECT 
first sketch

basic neuron class (main)

new and own approach to ML

author: T. Kowalski 
"""

# import sys
try:
    import cupy as cp
except ImportError:
    import numpy as cp


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
                serotonin_i: int = 0, serotonin_o: int = 0, callback_val=None
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
        
        self.callback=callback_val
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
            self.Q[i,:]=sigmoid(((data["output"]-data["input"])**2-callback_value**2)**3) # this approach means that neuron should amplify data...
            data["input"]*=self.Q[i,0]
            data["output"]*=self.Q[i,1]
            i+=1
    def threshold(self):
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
        Perform backpropagation to update quality factors (Q) based on the error.
        Args:
            target_value: The desired output value (callback_val).
        """
        learning_rate = 0.01  # Define a learning rate for adjustments

        i = 0
        for nt, data in self.neurotransmitters.items():
            # Calculate the error: difference between output and target
            output_error = data["output"] - target_value
            
            # Compute gradients for Q based on the error
            dQ_input = 2 * output_error * data["input"]  # Gradient w.r.t input
            dQ_output = 2 * output_error * data["output"]  # Gradient w.r.t output

            # Update Q using gradients
            self.Q[i, 0] -= learning_rate * dQ_input
            self.Q[i, 1] -= learning_rate * dQ_output
            
            # Clip Q values to avoid extreme changes (optional)
            self.Q = cp.clip(self.Q, 0.1, 10.0)

            i += 1

        
        
        # 
        #                   ======================================
        #                              Sample usage trial
        #                   ======================================
        # 

# assume we have only 1 sensor
sensor_data=1.0
callback_val=1.1
#   declaration of neurons - sample usage
module1=Neuron(acetylcholine_i=1,dopamine_o=2,acetylcholine_o=2,opioid_o=1,norepinephrine_o=1,callback_val=callback_val)
module2=Neuron(opioid_i=1,dopamine_i=1,acetylcholine_i=1,dopamine_o=1,opioid_o=1,callback_val=callback_val)
module3=Neuron(norepinephrine_i=1,dopamine_i=1,dopamine_o=1,acetylcholine_i=1,acetylcholine_o=1,opioid_o=1,callback_val=callback_val)
module_final=Neuron(opioid_i=2,dopamine_i=2,acetylcholine_i=1,dopamine_o=1,callback_val=callback_val)
# Connections now, yeah - i'll make it easier, maybe someday... all inside loop

print(sensor_data)


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
    sensor_data = 0.5 * out + 0.5 * callback_val

    # Backpropagation for each module to minimize the error
    module1.backpropagate(callback_val)
    module2.backpropagate(callback_val)
    module3.backpropagate(callback_val)
    module_final.backpropagate(callback_val)

    module1.threshold()
    module2.threshold()
    module3.threshold()
    module_final.threshold()

    print(i, sensor_data, callback_val, out)
    if i % 20 == 0:
        callback_val += 10
