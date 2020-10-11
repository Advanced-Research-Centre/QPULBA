from qiskit import QuantumCircuit
circ = QuantumCircuit(1)

import numpy as np
# circ.initialize([1, 1, 0, 0] / np.sqrt(2), [0, 1])
circ.ry(np.pi/2,0)

from qiskit import Aer, execute
simulator = Aer.get_backend('statevector_simulator')
sim_res = execute(circ, simulator).result()
statevector = sim_res.get_statevector(circ)
for i in statevector:
	print(i)