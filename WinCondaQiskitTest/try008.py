from qiskit import QuantumCircuit
import numpy as np
import random
from qiskit import Aer, execute
import math


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

a2 = np.pi * random.random()
a3 = np.pi * random.random()

# NEW

circ1 = QuantumCircuit(4)
#display(circ1,"all zero")

circ1.ry(a2,1)
circ1.ry(a3,2)
#display(circ1,"load fsm")

circ1.x(0)
circ1.toffoli(0,1,3)
circ1.x(0)
circ1.toffoli(0,2,3)
#display(circ1,"step 1")

circ1.swap(0,3)
#display(circ1,"reset 1")

circ1.x(0)
circ1.toffoli(0,1,3)
circ1.x(0)
circ1.toffoli(0,2,3)
#display(circ1,"step 2")

circ1.swap(0,3)
circ1.toffoli(0,2,3)
circ1.x(0)
circ1.toffoli(0,1,3)
circ1.x(0)
display(circ1,"reset 2")

print(circ1.draw())

circ2 = QuantumCircuit(5)
#display(circ2,"all zero")

circ2.ry(a2,1)
circ2.ry(a3,2)
#display(circ2,"load fsm")

circ2.x(0)
circ2.toffoli(0,1,3)
circ2.x(0)
circ2.toffoli(0,2,3)
#display(circ2,"step 1")

circ2.swap(0,3)
#display(circ2,"reset 1")

circ2.x(0)
circ2.toffoli(0,1,4)
circ2.x(0)
circ2.toffoli(0,2,4)
#display(circ2,"step 2")

circ2.swap(0,4)
display(circ2,"reset 2")

print(circ2.draw())