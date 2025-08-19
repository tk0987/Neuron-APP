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

        RECEPTOR_RULES = {
            "AMPA": {"stimuli": ["NMDA", "D1", "5-HT2A", "alpha1A"], "inhibitory": ["GABA_A", "GABA_B", "CB1", "D2"]},
            "NMDA": {"stimuli": ["AMPA", "D1", "5-HT2A", "H2"], "inhibitory": ["GABA_A", "5-HT1A", "CB1", "sigma1", "GlyR alpha1"]},
            "Kainate": {"stimuli": ["AMPA", "mGluR1"], "inhibitory": ["GABA_A"]},
            "mGluR1": {"stimuli": ["NMDA", "5-HT2A"], "inhibitory": ["GABA_B"]},
            "mGluR2": {"inhibitory": ["Glutamate release"]},
            "GABA_A": {"stimuli": ["5-HT1A", "D2", "Mu (mi)"], "inhibitory": ["AMPA", "NMDA", "H1", "NK1"]},
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
            "Mu (mi)": {"stimuli": ["GABA_A"], "inhibitory": ["NMDA", "5-HT2A"]},
            "GlyR alpha1": {"inhibitory": ["NMDA"]},
            "alpha1A": {"stimuli": ["AMPA"]},
            "alpha2A": {"inhibitory": ["D1", "5-HT2A"]},
            "sigma1": {"inhibitory": ["NMDA"]},
            "MT1": {"inhibitory": ["H1"]},
            "OX1": {"stimuli": ["Histamine release"]},
            "NK1": {"inhibitory": ["GABA_A"]}
        }

So, we have 14 main networks just in one neuron, and available subnets are the receptor types number: 92. Those perceptrons are interconnected inside of neuron (node) WITH STRICT RULES. Example: noradrenaline (alpha receptors):

        "alpha1A": {"stimuli": ["AMPA"]},
        "alpha2A": {"inhibitory": ["D1", "5-HT2A"]},
            
Stimuli and inhibitory means that the floating point numbers, inputs of noradrenaline network will be increased by ampa, and decreased by d1 and 5-ht2a - not only activation function is taken into the count.

So: assume, alpha1A has 2.0 on input. Theres also AMPA with value of 3, and D1 with value of 4. It means, that norepinephrine final output will be smaller (!) than after simple activation function - D1 (inhibitor) has bigger value than AMPA stimuli.

Therefore - each single neuron can act as XOR gate (and even more).

Whats important - the degree of influence by other networks adjusts during training.

Each neuron has memory, denoted by 'history' variable. History is just a list of previous transmitter values, and it take part in learning process, introducing hysteresis.
In real world cells, memory can be at DNA level - for example if g-protein activated certain genes, altering transmitter production at output. It can be also increase/decrease in receptor numbers at input, due to sensitization/tolerance developement. 

Here number of receptors is stable, but their response differs. It would be nice to code net with floating internal structure over time - not predefined like here. I hope I ll do it some day.

Loss is very important - it is the info about difference between desired state and the goal to achieve. Just like in robots - Loss can be just the difference between sensor reading (current) and the desired state. The net 'll learn to achieve that. To compensate non-desired state of the system.

3. HOW TO USE THIS

Make complex networks. 

In fact, this is under development now - I cannot tell exactly how to use this project (i mean technically). Just use it this way, that it works. Well...

this static neuron.Neuron.connect_neurons() method is forming connections kinda like connections between two tf.keras.layers.Dense(). It connects all-with-all.

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
