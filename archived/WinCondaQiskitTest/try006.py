from qiskit import QuantumCircuit
import numpy as np
import random
from qiskit import Aer, execute

qubits = 17
circ = QuantumCircuit(qubits)

simulator = Aer.get_backend('statevector_simulator')
def display(circ):
	sim_res = execute(circ, simulator).result()
	statevector = sim_res.get_statevector(circ)
	print("============ State Vector ============")
	for i in statevector:
		print(i)	
	print("============..............============")

# initialize
for q in range(0,qubits):
	ang = np.pi * random.random()
	circ.ry(ang,q)
display(circ)