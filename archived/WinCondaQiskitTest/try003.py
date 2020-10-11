#from qiskit import QuantumRegister, ClassicalRegister
#from qiskit.tools.visualization import plot_histogram, plot_state_city

from qiskit import QuantumCircuit
circ = QuantumCircuit(2, 2)

import numpy as np
circ.initialize([1, 1, 0, 0] / np.sqrt(2), [0, 1])

circ.h(0)
circ.cx(0, 1)
circ.measure([0,1], [0,1])

from qiskit import Aer
# print(Aer.backends())
# 'qasm_simulator', 'statevector_simulator', 'unitary_simulator', 'pulse_simulator'

sim1 = Aer.get_backend('qasm_simulator')

from qiskit import execute
result1 = execute(circ, sim1, shots=10, memory=True).result()
counts = result1.get_counts(circ)
print(counts)
# memory = result1.get_memory(circ)
# print(memory)

sim2 = Aer.get_backend('statevector_simulator')
result2 = execute(circ, sim2).result()
statevector = result2.get_statevector(circ)
print(statevector)