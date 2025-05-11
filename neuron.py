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
    def __init__(self, 
                 norepinephrine_i=0, norepinephrine_o=0,
                 acetylcholine_i=0, acetylcholine_o=0,
                 dopamine_i=0, dopamine_o=0,
                 gaba_i=0, gaba_o=0,
                 glutamate_i=0, glutamate_o=0,
                 glycine_i=0, glycine_o=0,
                 histamine_i=0, histamine_o=0,
                 opioid_i=0, opioid_o=0,
                 serotonin_i=0, serotonin_o=0, 
                 callback_val=None,
                 recurrent=False):
        
        self.Q = cp.ones(shape=(10, 2))  # quality factors
        self.callback = callback_val
        self.recurrent = recurrent
        self._prev_outputs = {}

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

        # Initialize arrays
        for nt, counts in self.neurotransmitters.items():
            setattr(self, f"{nt}_i", cp.zeros(counts["input"]))
            setattr(self, f"{nt}_o", cp.zeros(counts["output"]))

        # Validation
        if any(cp.any(getattr(self, f"{nt}_i")) < 0 or cp.any(getattr(self, f"{nt}_o")) < 0 for nt in self.neurotransmitters):
            raise ValueError("Input/output counts must be non-negative.")
        
        # Logical dependencies
        if acetylcholine_i + acetylcholine_o > 0 and dopamine_i + dopamine_o == 0:
            raise ValueError("Acetylcholine requires dopamine.")
        if gaba_i + gaba_o > 0 and (
            dopamine_i + dopamine_o == 0 or
            glutamate_i + glutamate_o == 0 or
            acetylcholine_i + acetylcholine_o == 0):
            raise ValueError("GABA requires dopamine, acetylcholine, or glutamate.")
        if serotonin_i + serotonin_o > 0 and dopamine_i + dopamine_o == 0:
            raise ValueError("Serotonin requires dopamine.")
        if glycine_i + glycine_o > 0 and acetylcholine_i + acetylcholine_o == 0:
            raise ValueError("Glycine requires acetylcholine.")

    def autolinker(self, target_neuron, mapping: dict):
        for nt, (from_idx, to_idx) in mapping.items():
            try:
                val = getattr(self, f"{nt}_o")[from_idx]
                getattr(target_neuron, f"{nt}_i")[to_idx] = val
            except (AttributeError, IndexError) as e:
                print(f"[autolinker] Failed linking {nt} from {from_idx} to {to_idx}: {e}")

    def inject_recurrence(self):
        if self.recurrent and self._prev_outputs:
            for nt in self.neurotransmitters:
                prev_out = self._prev_outputs.get(nt)
                curr_in = getattr(self, f"{nt}_i", None)
                if prev_out is not None and curr_in is not None:
                    min_len = min(len(prev_out), len(curr_in))
                    curr_in[:min_len] += prev_out[:min_len]

    def response(self):
        rules = {
            "norepinephrine": {"stimuli": ["acetylcholine", "dopamine", "glutamate", "serotonin", "histamine"], "inhibitory": ["gaba", "opioid", "glycine"]},
            "acetylcholine": {"stimuli": ["dopamine", "glutamate", "serotonin", "norepinephrine", "histamine"], "inhibitory": ["gaba", "glycine", "opioid"]},
            "dopamine": {"stimuli": ["glutamate", "norepinephrine", "histamine", "opioid"], "inhibitory": ["gaba", "acetylcholine", "serotonin", "glycine"]},
            "gaba": {"stimuli": ["dopamine", "glutamate", "glycine", "opioid"], "inhibitory": ["acetylcholine"]},
            "glutamate": {"stimuli": ["acetylcholine", "dopamine", "glycine"], "inhibitory": ["gaba", "opioid"]},
            "glycine": {"stimuli": ["gaba", "glutamate"], "inhibitory": ["dopamine", "acetylcholine"]},
            "histamine": {"stimuli": ["dopamine", "glutamate", "acetylcholine"], "inhibitory": ["gaba"]},
            "opioid": {"stimuli": ["dopamine", "gaba"], "inhibitory": ["glutamate", "serotonin"]},
            "serotonin": {"stimuli": ["acetylcholine", "dopamine", "norepinephrine"], "inhibitory": ["gaba"]}
        }

        for nt in self.neurotransmitters:
            stimuli = sum(cp.sum(getattr(self, f"{stim}_i", 0)) for stim in rules[nt]["stimuli"])
            inhibitory = sum(cp.sum(getattr(self, f"{inh}_i", 0)) for inh in rules[nt]["inhibitory"])
            response = stimuli - inhibitory
            if getattr(self, f"{nt}_o", None) is not None:
                getattr(self, f"{nt}_o")[:] = response

        if self.recurrent:
            for nt in self.neurotransmitters:
                self._prev_outputs[nt] = getattr(self, f"{nt}_o").copy()

    def optimizer(self, callback_value):
        def sigmoid(value):
            return 1.0 / (1.0 + cp.exp(-1.0 * value)) + 1.0

        for i, (nt, data) in enumerate(self.neurotransmitters.items()):
            self.Q[i, :] = sigmoid(((data["output"] - data["input"]) ** 2 - callback_value ** 2) ** 3)
            data["input"] *= self.Q[i, 0]
            data["output"] *= self.Q[i, 1]

    def threshold(self):
        self.thresholds = cp.ones_like(self.Q)
        for i, (nt, data) in enumerate(self.neurotransmitters.items()):
            self.thresholds[i, 0] *= data["input"]
            self.thresholds[i, 1] *= data["output"]
        self.optimizer(self.callback)
        for i, (nt, data) in enumerate(self.neurotransmitters.items()):
            if data["input"] > self.thresholds[i, 0]:
                self.response()
            else:
                data["output"] = 0.0

    def backpropagate(self, target_value):
        learning_rate = 0.01
        for i, (nt, data) in enumerate(self.neurotransmitters.items()):
            output_error = data["output"] - target_value
            dQ_input = 2 * output_error * data["input"]
            dQ_output = 2 * output_error * data["output"]
            self.Q[i, 0] -= learning_rate * dQ_input
            self.Q[i, 1] -= learning_rate * dQ_output
        self.Q = cp.clip(self.Q, 0.1, 10.0)


# =============================
#          SAMPLE USAGE
# =============================

sensor_data = 1.0
callback_val = 1.1

module1 = Neuron(acetylcholine_i=1, dopamine_o=2, acetylcholine_o=2, opioid_o=1, norepinephrine_o=1, callback_val=callback_val)
module2 = Neuron(opioid_i=1, dopamine_i=1, acetylcholine_i=1, dopamine_o=1, opioid_o=1, callback_val=callback_val)
module3 = Neuron(norepinephrine_i=1, dopamine_i=1, dopamine_o=1, acetylcholine_i=1, acetylcholine_o=1, opioid_o=1, callback_val=callback_val, recurrent=True)
module_final = Neuron(opioid_i=2, dopamine_i=2, acetylcholine_i=1, dopamine_o=1, callback_val=callback_val)

print(sensor_data)

for i in range(1000):
    module1.acetylcholine_i[0] = sensor_data
    module1.response()

    module1.autolinker(module2, {
        "dopamine": (0, 0),
        "opioid": (0, 0),
        "acetylcholine": (0, 0),
    })

    module1.autolinker(module3, {
        "norepinephrine": (0, 0),
        "dopamine": (1, 0),
        "acetylcholine": (1, 0),
    })

    module2.response()
    module3.inject_recurrence()
    module3.response()

    module2.autolinker(module_final, {
        "opioid": (0, 0),
        "dopamine": (0, 1),
    })
    module3.autolinker(module_final, {
        "opioid": (0, 1),
        "dopamine": (0, 0),
        "acetylcholine": (0, 0),
    })

    module_final.inject_recurrence()
    module_final.response()
    out = module_final.dopamine_o[0]
    sensor_data = 0.5 * out + 0.5 * callback_val

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
