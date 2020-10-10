from qiskit import QuantumCircuit
import numpy as np
import random
from qiskit import Aer, execute
import math

circ = QuantumCircuit(4)

simulator = Aer.get_backend('statevector_simulator')
def display(circ,msg=""):
	sim_res = execute(circ, simulator).result()
	statevector = sim_res.get_statevector(circ)
	qb = int(math.log2(len(statevector)))
	print("============ State Vector ============", msg)
	s = 0
	for i in statevector:
		print('  ({:.5f})   |{:0{}b}>'.format(i,s,qb))
		s = s+1
	print("============..............============")

# circ.initialize([1, 1, 0, 0] / np.sqrt(2), [0, 1])
display(circ,"step 0")

# initialize
a1 = np.pi * random.random()
circ.ry(a1,0)
display(circ,"step 1")

a2 = np.pi * random.random()
circ.ry(a2,1)
display(circ,"step 2")

a3 = np.pi * random.random()
#circ.ry(a3,2)
display(circ,"step 3")

circ.x(0)
circ.toffoli(0,1,3)
circ.x(0)
circ.toffoli(0,2,3)
display(circ,"step 4")

print(circ.draw())