import numpy as np
import random
from qiskit import QuantumCircuit, Aer, execute
from math import log2, ceil, pi, sin
from numpy import savetxt, save, savez_compressed
import sys

#=====================================================================================================================

simulator = Aer.get_backend('statevector_simulator')

def disp_isv(circ, msg="", all=True, precision=1e-8):
	sim_res = execute(circ, simulator).result()
	statevector = sim_res.get_statevector(circ)
	qb = int(log2(len(statevector)))
	print("\n============ State Vector ============", msg)
	s = 0
	for i in statevector:
		if (all == True): print('  ({:+.5f})   |{:0{}b}>'.format(i,s,qb))
		else:
			if (abs(i) > precision): print('  ({:+.5f})   |{:0{}b}>'.format(i,s,qb))
		s = s+1
	print("============..............============")
	return

#=====================================================================================================================

fsm = [0,1,2,3]
tape = [4,5,6,7]
ancilla = [8]
count = [9,10,11,12,13,14]

searchbits = 5
for j in range(4,7):
	print("\nDetectable solutions with %d count bits:",j)
	countbits = j
	for i in range(0,countbits**2):
		theta = (i/(2**countbits))*pi*2
		counter = 2**searchbits * (1 - sin(theta/2)**2)
		print(round(counter),"|",end='')
# sys.exit(0)

qcirc_width = len(fsm) + len(tape) + len(ancilla) + len(count)
qcirc = QuantumCircuit(qcirc_width, len(count))

# U_qpulba121
# qcirc.h(fsm)
# qcirc.cx(fsm[1], ancilla[0])
# qcirc.cx(fsm[0],tape[0])
# qcirc.cx(fsm[0],tape[1])
# qcirc.cx(fsm[0],tape[2])
# qcirc.cx(fsm[0],tape[3])

qcirc.h(tape)
qcirc.cx(tape[1], ancilla[0])   # Gives 32 solns: wrong
# qcirc.h(ancilla[0])   # Gives 2 solns: correct

disp_isv(qcirc, "Step: Run QPULBA 121", all=False, precision=1e-4)
# sys.exit(0)

#=====================================================================================================================

def condition_fsm(qcirc, fsm, tape):
	# Finding specific programs-output characteristics			(fsm|tape)
	# e.g. Self-replication
	for q in fsm:
		qcirc.cx(q,tape[q])
	qcirc.barrier()
	return

#=====================================================================================================================

# search = tape
# condition_fsm(qcirc, fsm, tape)

# disp_isv(qcirc, "Step: Find self-replicating programs", all=False, precision=1e-4)
# sys.exit(0)

#=====================================================================================================================

def U_oracle(sz):
	# Mark fsm/tape/state with all zero Hamming distance (matches applied condition perfectly)
	tgt_reg = list(range(0,sz))
	oracle = QuantumCircuit(len(tgt_reg))
	oracle.x(tgt_reg)
	oracle.h(tgt_reg[0])
	oracle.mct(tgt_reg[1:],tgt_reg[0])
	oracle.h(tgt_reg[0])
	oracle.x(tgt_reg)
	return oracle

def U_diffuser(sz):
    # https://qiskit.org/textbook/ch-algorithms/quantum-counting.html
    tgt_reg = list(range(0,sz))
    diffuser = QuantumCircuit(len(tgt_reg))
    diffuser.h(tgt_reg[1:])
    diffuser.x(tgt_reg[1:])
    diffuser.z(tgt_reg[0])
    diffuser.mct(tgt_reg[1:],tgt_reg[0])
    diffuser.x(tgt_reg[1:])
    diffuser.h(tgt_reg[1:])
    diffuser.z(tgt_reg[0])
    return diffuser

def U_QFT(n):
    # n-qubit QFT circuit
    qft = QuantumCircuit(n)
    def swap_registers(qft, n):
        for qubit in range(n//2):
            qft.swap(qubit, n-qubit-1)
        return qft
    def qft_rotations(qft, n):
        # Performs qft on the first n qubits in circuit (without swaps)
        if n == 0:
            return qft
        n -= 1
        qft.h(n)
        for qubit in range(n):
            qft.cu1(np.pi/2**(n-qubit), qubit, n)
        qft_rotations(qft, n)
    qft_rotations(qft, n)
    swap_registers(qft, n)
    return qft

#=====================================================================================================================

selregs = [4,5,6,7,8]
# selregs = [4,5,6,7]
# selregs = [0,1,2,3,4,5,6,7,8]

# Create controlled Grover oracle circuit
# oracle = U_oracle(len(search)).to_gate()
oracle = U_oracle(len(selregs)).to_gate()
c_oracle = oracle.control()
c_oracle.label = "cGO"

# Create controlled Grover diffuser circuit
diffuser = U_diffuser(len(selregs)).to_gate()
c_diffuser = diffuser.control()
c_diffuser.label = "cGD"

# Create inverse QFT circuit
iqft = U_QFT(len(count)).to_gate().inverse()
iqft.label = "iQFT"

#=====================================================================================================================

qcirc.h(count)
qcirc.barrier()

# Begin controlled Grover iterations
iterations = 1
for qb in count:
	for i in range(iterations):
		# qcirc.append(c_oracle, [qb] + search)
		qcirc.append(c_oracle, [qb] + selregs)
		qcirc.append(c_diffuser, [qb] + selregs)
	iterations *= 2
	qcirc.barrier()

# Inverse QFT
qcirc.append(iqft, count)
qcirc.barrier()

# Measure counting qubits
qcirc.measure(count, range(len(count)))

# print(qcirc.draw())
# sys.exit(0)

#=====================================================================================================================

emulator = Aer.get_backend('qasm_simulator')
job = execute(qcirc, emulator, shots=1024)
hist = job.result().get_counts()
# print(hist)

measured_int = int(max(hist, key=hist.get),2)
theta = (measured_int/(2**len(count)))*pi*2
# counter = 2**4 * (1 - sin(theta/2)**2)
# print("Number of solutions = %.1f" % counter)
counter = 2**len(selregs) * (1 - sin(theta/2)**2)
print("Number of solutions = %.1f" % counter)
		
#=====================================================================================================================