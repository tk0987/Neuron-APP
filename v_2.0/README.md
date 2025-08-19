# Neuron APP - Advanced Perceptron Project
ML tool for various tasks - to be continued

the goal is to script and still have good results + inference. along with to learn something new. one can even create neuron instance(s) and save is as binary - then load.

1. PROJECT DESCRIPTION

Perceptron much more biological neuron-like than perceptrons in current use. Within one graph node several networks are connected interacting with each other - within built-in strict rules. And yes...

One neuron from this project CAN be XOR gate.

Imagine, that you have just a small python board in application that process sensor data into project response. For example - autonomic vechicle. This ML tool should learn when to turn - without supervision of user.
It should be also light for very low inference times on uC. 

Just script, as you dont need binaries here or 'lite' versions of libraries (eg. tensorflow)

here is a trial - still under developement, but first results are promising.


2. WHAT IS INSIDE

Well. Each neuron is just like perceptrons - multiple just in one node. Those perceptrons inside one neuron can interact with each other, influencing/altering their action. Look:

    {
      "Glutamate": {
        "Ionotropic": ["AMPA", "NMDA", "Kainate"],
        "Metabotropic": ["mGluR1", "mGluR2", "mGluR3", "mGluR4", "mGluR5", "mGluR6", "mGluR7", "mGluR8"]
      },
      "GABA": {
        "Ionotropic": ["GABA_A"],
        "Metabotropic": ["GABA_B"]
      },
      "Dopamine": {
        "D1-like": ["D1", "D5"],
        "D2-like": ["D2", "D3", "D4"]
      },
      "Serotonin": {
        "Receptors": [
          "5-HT1A", "5-HT1B", "5-HT1D", "5-HT1E", "5-HT1F",
          "5-HT2A", "5-HT2B", "5-HT2C",
          "5-HT3",
          "5-HT4", "5-HT5A", "5-HT5B", "5-HT6", "5-HT7"
        ]
      },
      "Acetylcholine": {
        "Ionotropic": ["nAChR (α1–α10)", "nAChR (β1–β4)", "γ", "δ", "ε"],
        "Metabotropic": ["M1", "M2", "M3", "M4", "M5"]
      },
      "Histamine": {
        "Receptors": ["H1", "H2", "H3", "H4"]
      },
      "Norepinephrine": {
        "Alpha": ["α1A", "α1B", "α1D", "α2A", "α2B", "α2C"],
        "Beta": ["β1", "β2", "β3"]
      },
      "Opioid": {
        "Receptors": ["Mu (μ)", "Delta (δ)", "Kappa (κ)", "Nociceptin (ORL1)"]
      },
      "Glycine": {
        "Receptors": ["GlyR α1", "GlyR α2", "GlyR α3", "GlyR α4", "GlyR β"]
      },
      "Cannabinoid": {
        "Receptors": ["CB1", "CB2"]
      },
      "Purinergic": {
        "Ionotropic": ["P2X1", "P2X2", "P2X3", "P2X4", "P2X5", "P2X6", "P2X7"],
        "Metabotropic": ["P2Y1", "P2Y2", "P2Y4", "P2Y6", "P2Y11", "P2Y12", "P2Y13", "P2Y14"]
      },
      "Other": {
        "TRP": ["TRPV1"],
        "Sigma": ["σ1", "σ2"],
        "Melatonin": ["MT1", "MT2"],
        "Orexin": ["OX1", "OX2"],
        "Tachykinin": ["NK1", "NK2", "NK3"]
      }
    }  
These above are just main transmitters and their receptors - the first part to be aware of. Some of the receptors are altering function of others - negatively and positively, depending whether they are stimuli/inhibitory. Now the more funny part - the way, how all of those nets are encoded to interact with each other inside single AP - advanced perceptron:

    {
      "AMPA": {
        "stimuli": ["NMDA", "D1", "5-HT2A"],
        "inhibitory": ["GABA_A", "GABA_B"]
      },
      "NMDA": {
        "stimuli": ["AMPA", "D1", "5-HT2A"],
        "inhibitory": ["GABA_A", "5-HT1A"]
      },
      "Kainate": {
        "stimuli": ["AMPA", "mGluR1"],
        "inhibitory": ["GABA_A"]
      },
      "mGluR1": {
        "stimuli": ["NMDA", "5-HT2A"],
        "inhibitory": ["GABA_B"]
      },
      "GABA_A": {
        "stimuli": ["5-HT1A", "D2"],
        "inhibitory": ["AMPA", "NMDA", "H1"]
      },
      "GABA_B": {
        "stimuli": ["D2", "5-HT1B"],
        "inhibitory": ["AMPA", "mGluR1"]
      },
      "D1": {
        "stimuli": ["NMDA", "5-HT2A"],
        "inhibitory": ["GABA_A", "D2"]
      },
      "D2": {
        "stimuli": ["GABA_B", "5-HT1A"],
        "inhibitory": ["D1", "AMPA"]
      },
      "5-HT1A": {
        "stimuli": ["GABA_A", "D2"],
        "inhibitory": ["NMDA", "5-HT2A"]
      },
      "5-HT2A": {
        "stimuli": ["D1", "NMDA"],
        "inhibitory": ["GABA_A", "5-HT1A"]
      },
      "H1": {
        "stimuli": ["D1", "5-HT2C"],
        "inhibitory": ["GABA_A"]
      },
      "H2": {
        "stimuli": ["D2", "NMDA"],
        "inhibitory": ["GABA_B"]
      },
      "CB1": {
        "stimuli": ["GABA_B"],
        "inhibitory": ["NMDA", "AMPA"]
      },
      "Mu (μ)": {
        "stimuli": ["GABA_A"],
        "inhibitory": ["NMDA", "5-HT2A"]
      }
    }
              


So, we have 9 perceptrons just in one neuron. Those perceptrons are interconnected inside of neuron (node) WITH STRICT RULES. Example: noradrenaline:

    "norepinephrine": {
                    "stimuli": ["acetylcholine", "dopamine", "glutamate", "serotonin", "histamine"],
                    "inhibitory": ["gaba", "opioid", "glycine"]
                },
            
Stimuli means that the floating point numbers, inputs of noradrenaline network will be increased by ach, dopamine, glutamate, serotonin and histamine - not only activation function. Numbers values, carried in norepinephrine net, will be also downgraded by gaba, opioid and glycine networks values.
So: assume, norepinephrine has 2.0 on input. Theres also dopamine with value of 3, and gaba with value of 4. It means, that norepinephrine final output will be smaller (!) than after simple activation function - gaba (inhibitor) has bigger value than dopamine stimuli.

Therefore - each single neuron can act as XOR gate (and even more).

Whats important - the degree of influence by other networks adjusts during training. Same for threshold and activation - as output is thresholded.

Each neuron has memory, denoted by 'history' variable. History is just a list of previous transmitter values, and it take part in learning process, introducing hysteresis.
In real world cells, memory can be at DNA level - for example if g-protein activated certain genes, altering transmitter production at output. It can be also increase/decrease in receptor numbers at input, due to sensitization/tolerance developement. 

Here number of receptors is stable, but their response differs. It would be nice to code net with floating internal structure over time - not predefined like here. I hope I ll do it some day.

Callback_val is very important - it is the network goal to achieve. Just like in robots - callback_val can be desired sensor value. The net 'll learn to achieve that.

3. HOW TO USE THIS

Make complex networks. 

In fact, this is under development now - I cannot tell exactly how to use this project (i mean technically). Just use it this way, that it works. Well...

Example net:

    module1=Neuron(acetylcholine_i=1,dopamine_o=2,acetylcholine_o=2,opioid_o=1,norepinephrine_o=1,callback_val=callback_val)
    module2=Neuron(opioid_i=1,dopamine_i=1,acetylcholine_i=1,dopamine_o=1,opioid_o=1,callback_val=callback_val)
    module3=Neuron(norepinephrine_i=1,dopamine_i=1,dopamine_o=1,acetylcholine_i=1,acetylcholine_o=1,opioid_o=1,callback_val=callback_val)
    module_final=Neuron(opioid_i=2,dopamine_i=2,acetylcholine_i=1,dopamine_o=1,callback_val=callback_val)

so - 4 neurons, with different nets inside them. How we can join them manually? Example of manual connecting nets:

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

As it is - we can connect them receptor-wise (like above, two neuron nodes are forming something like 'middle layer' of the full network - with one input layer, and one output layer.

This was init and connection of modules. How to use them, that they can optimize themselves?

    # Backpropagation for each module to minimize the error
    module1.backpropagate(callback_val)
    module2.backpropagate(callback_val)
    module3.backpropagate(callback_val)
    module_final.backpropagate(callback_val)

    module1.threshold()
    module2.threshold()
    module3.threshold()
    module_final.threshold()

So - for learning process: 
1. Backpropagate, then
2. Threshold

I think I ll use backpropagate only, fusing threshold() method into it.

Another example, with semi-automatic input-output connections, and the init of neurons:

    first_layout={}
    for _ in range(6):
        first_layout[f"var1_{_}"] = n.Neuron(norepinephrine_i=1,norepinephrine_o=1,acetylcholine_i=5,acetylcholine_o=6,gaba_i=5,gaba_o=6,opioid_o=2,history=12)
    second_layout={}
    for _ in range(12):
        second_layout[f"var2_{_}"]=n.Neuron(norepinephrine_i=1,norepinephrine_o=4,glutamate_i=6,glutamate_o=12,acetylcholine_i=6, \ 
                                            acetylcholine_o=6,gaba_i=6,gaba_o=12,opioid_i=2,opioid_o=4,history=24)
    third_layout={}
    for _ in range(6):
        third_layout[f"var3_{_}"] = n.Neuron(norepinephrine_i=4,glutamate_i=12,glutamate_o=6,acetylcholine_i=6,gaba_i=6,opioid_i=4,history=56)
    
    # input and output:
    inputs = []
    outputs = []
    # inps
    for i in range(6):
        neuron = first_layout[f"var1_{i}"]
        for nt in neuron.neurotransmitters:
            input_array = getattr(neuron, f"{nt}_i", None)
            if input_array is not None:
                inputs.append(input_array)
    # outs
    for i in range(6):
        neuron = third_layout[f"var1_{i}"]
        for nt in neuron.neurotransmitters:
            output_array = getattr(neuron, f"{nt}_i", None)
            if output_array is not None:
                outputs.append(output_array)

If you dont want to manually connect them, you can use static method built-in Neuron() class - neuron.Neuron.connect_neurons(), like here:

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
            for nt in third_layout[f"var3_{i}"].neurotransmitters:
                n.Neuron.connect_neurons(
                    third_layout[f"var3_{i}"], second_layout[f"var2_{j}"], nt
                )

this static neuron.Neuron.connect_neurons() method is forming connections kinda like connections between two tf.keras.layers.Dense(). It connects all-with-all, but only transmitter-wise, so opioid cannot be connected with gaba.

Of course, you can manually connect e.g. dopamine with serotonine, but... it is heresy. Do not do that - it will work, most probably, but keep it clean, please!

    # ============+++++++++++++++++++++++================
    #
    #                WHEN TO USE THIS
    #
    # ============+++++++++++++++++++++++================

Yes - the most important part, something so vivid (for the author only), that I forgot to write about this.

Network returns compensative response for input - using loss. It returns a change which is needed to improve current state.

It is well-suited for sensor reading compensation, like in autonomic machines. Most probably it could be used for denoising, encoding, and so on - but, really - it is hard thing to work with. It needs time.

For example you can compute loss as a discrepancy between desired state (drone oriented horizontally, accel shows acceleration down) and current state - yaw, pitch, roll. And the net will return a value of the change needed for acquiring desired state in respect for motor/servo action it learned - after several iterations its performance will be only better.
