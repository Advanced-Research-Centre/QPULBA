from qiskit import QuantumCircuit
import numpy as np
import random
from qiskit import Aer, execute

circ = QuantumCircuit(4)

simulator = Aer.get_backend('statevector_simulator')
def display(circ):
	sim_res = execute(circ, simulator).result()
	statevector = sim_res.get_statevector(circ)
	print("============ State Vector ============")
	for i in statevector:
		print(i)	
	print("============..............============")

# circ.initialize([1, 1, 0, 0] / np.sqrt(2), [0, 1])
display(circ)

# initialize
a1 = np.pi * random.random()
circ.ry(a1,0)
a2 = np.pi * random.random()
circ.ry(a2,1)
a3 = np.pi * random.random()
circ.ry(a3,2)
display(circ)