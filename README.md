# neuron-ML-tool
ML tool for various tasks - to be continued

the goal is to script, not compile and have good results. along with to learn something new

# PROJECT DESCRIPTION

Imagine, that you have just a small python board in application that process sensor data into project response. For example - autonomic vechicle. This ML tool should learn when to turn - without supervision of user.
It should be also light for very low inference times on uC. 

Just script, as you dont need binaries here or 'lite' versions of libraries (eg. tensorflow)

here is a trial - still under developement, but first results are promising.


please note, that network complexity can be very high.

One can make many networks in one code, and this networks could act independently - like fpga

# todo:

1. cleanup and tide,
2. tolerance/sensitization for thresholding [? - or not],
3. make this on lists without this incredible numpy - for uC
