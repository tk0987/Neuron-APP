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

    def __init__(self, 
                norepinephrine_i: int = 0, norepinephrine_o: int = 0,
                acetylcholine_i: int = 0, acetylcholine_o: int = 0,
                dopamine_i: int = 0, dopamine_o: int = 0,
                gaba_i: int = 0, gaba_o: int = 0,
                glutamate_i: int = 0, glutamate_o: int = 0,
                glycine_i: int = 0, glycine_o: int = 0,
                histamine_i: int = 0, histamine_o: int = 0,
                opioid_i: int = 0, opioid_o: int = 0,
                serotonin_i: int = 0, serotonin_o: int = 0, callback_val=None, history: int = 16
                ):

So, we have 9 perceptrons just in one neuron. Those perceptrons are interconnected inside of neuron (node) WITH STRICT RULES. Example: noradrenaline:

    "norepinephrine": {
                    "stimuli": ["acetylcholine", "dopamine", "glutamate", "serotonin", "histamine"],
                    "inhibitory": ["gaba", "opioid", "glycine"]
                },
            
Stimuli means that the floating point numbers, inputs of noradrenaline network will be increased by ach, dopamine, glutamate, serotonin and histamine - not only activation function. Their value will be also downgraded by gaba, opioid and glycine networks values.
So: assume, norepinephrine has 2.0 on input. Theres also dopamine with value of 3, and gaba with value of 4. It means, that norepinephrine final output will be smaller (!) than after simple activation function - gaba (inhibitor) has bigger value than dopamine stimuli.

Therefore - each single neuron can act as XOR gate (and even more).

Whats important - the degree of influence by other networks adjusts during training. Same for threshold for activation - as it is thresholded.

Each neuron has memory, denoted by 'history' variable. History is just a list of previous transmitter values, and it take part in learning process, introducing hysteresis.

Callback_val is very important - it is the network goal to achieve. Just like in robots - callback_val can be desired sensor value. The net 'll learn to achieve that.

3. HOW TO USE THIS

Make complex networks. 

In fact, this is under development now - I cannot tell exactly how to use this project (i mean technically). Just use it this way, that it works. Well...
