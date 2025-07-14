import neuron as n
try:
    import cupy as np
except ImportError:
    import numpy as np
    
    
    """
just in case... no comments, it is scratch as for now
    
    
    """

callback_val=0 # to be added from data


first_layout={}
for _ in range(6):
    first_layout[f"var1_{_}"] = n.Neuron(norepinephrine_i=1,norepinephrine_o=1,acetylcholine_i=5,acetylcholine_o=6,gaba_i=5,gaba_o=6,opioid_o=2)
second_layout={}
for _ in range(12):
    second_layout[f"var2_{_}"]=n.Neuron(norepinephrine_i=1,norepinephrine_o=4,glutamate_i=6,glutamate_o=12,acetylcholine_i=6,acetylcholine_o=6,gaba_i=6,gaba_o=12,opioid_i=2,opioid_o=4)
third_layout={}
for _ in range(6):
    third_layout[f"var3_{_}"] = n.Neuron(norepinephrine_i=4,glutamate_i=12,glutamate_o=6,acetylcholine_i=6,gaba_i=6,opioid_i=4)
    

for i in range(6):
    for j in range(12):
        for nt in first_layout[f"var1_{i}"].neurotransmitters:
            n.Neuron.connect_neurons(
                first_layout[f"var1_{i}"], second_layout[f"var2_{j}"], nt
            )
for i in range(12):
    for j in range(6):
        for nt in second_layout[f"var2_{i}"].neurotransmitters:
            n.Neuron.connect_neurons(
                second_layout[f"var2_{i}"], third_layout[f"var3_{j}"], nt
            )

for i in range(6):
    for j in range(12):
        for nt in third_layout[f"var1_{i}"].neurotransmitters:
            n.Neuron.connect_neurons(
                third_layout[f"var1_{i}"], second_layout[f"var2_{j}"], nt
            )
# this will be training...

for neuron in first_layout.values():
    neuron.backpropagate(callback_val)
    neuron.threshold()

for neuron in second_layout.values():
    neuron.backpropagate(callback_val)
    neuron.threshold()

for neuron in third_layout.values():
    neuron.backpropagate(callback_val)
    neuron.threshold()
