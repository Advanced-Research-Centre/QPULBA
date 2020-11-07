import numpy as np
import random
from qiskit import QuantumCircuit, Aer, execute
from math import log2, ceil, pi, sin

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

def U_oracle1(sz):
    # Mark fsm/tape/state with all zero Hamming distance (matches applied condition perfectly)
    tgt_reg = list(range(0,sz))
    oracle = QuantumCircuit(len(tgt_reg))
    oracle.x(tgt_reg)
    oracle.h(tgt_reg[0])
    oracle.mct(tgt_reg[1:],tgt_reg[0])
    oracle.h(tgt_reg[0])
    oracle.x(tgt_reg)
    return oracle

def U_oracle5(sz):
    # Mark {0111, 1111, 1001, 1011, 0101}
    tgt_reg = list(range(0,sz))
    oracle = QuantumCircuit(len(tgt_reg))
    oracle.h([2,3])
    oracle.ccx(0,1,2)
    oracle.h(2)
    oracle.x(2)
    oracle.ccx(0,2,3)
    oracle.x(2)
    oracle.h(3)
    oracle.x([1,3])
    oracle.h(2)
    oracle.mct([0,1,3],2)
    oracle.x([1,3])
    oracle.h(2)
    return oracle

#=====================================================================================================================

# def U_diffuser(sz):
# 	# Amplitude amplification on all qubits except count
# 	tgt_reg = list(range(0,sz))
# 	diffuser = QuantumCircuit(len(tgt_reg))
# 	diffuser.h(tgt_reg)
# 	diffuser.x(tgt_reg)
# 	diffuser.h(tgt_reg[0])
# 	diffuser.mct(tgt_reg[1:],tgt_reg[0])
# 	diffuser.h(tgt_reg[0])
# 	diffuser.x(tgt_reg)
# 	diffuser.h(tgt_reg)
# 	return diffuser

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

#=====================================================================================================================

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

qnos = [1, 4, 4]

dummy	= list(range(sum(qnos[0:0]),sum(qnos[0:1])))
search	= list(range(sum(qnos[0:1]),sum(qnos[0:2])))
count	= list(range(sum(qnos[0:2]),sum(qnos[0:3])))

qcirc_width = sum(qnos[0:3])
qcirc = QuantumCircuit(qcirc_width, len(count))

#=====================================================================================================================

# Create controlled Grover oracle circuit
oracle = U_oracle1(len(search)).to_gate()
c_oracle = oracle.control()
c_oracle.label = "cGO"

# Create controlled Grover diffuser circuit
diffuser = U_diffuser(len(search)).to_gate()
c_diffuser = diffuser.control()
c_diffuser.label = "cGD"

# Create inverse QFT circuit
iqft = U_QFT(len(count)).to_gate().inverse()
iqft.label = "iQFT"

#=====================================================================================================================

# product state qubits not part of search qubits does not matter even if superposed
# qcirc.i(dummy)
# qcirc.x(dummy)
# qcirc.h(dummy) 
# qcirc.ry(0.25,dummy)

# qcirc.ry(0.55,search) # probability  of states assumed to be equal in counting
qcirc.h(search)
qcirc.barrier()

print()
disp_isv(qcirc, "Step: Search state vector", all=False, precision=1e-4)

qcirc.h(count)
qcirc.barrier()

# Begin controlled Grover iterations
iterations = 1
for qb in count:
    for i in range(iterations):
        qcirc.append(c_oracle, [qb] + search)
        qcirc.append(c_diffuser, [qb] + search)
    iterations *= 2
    qcirc.barrier()

# Inverse QFT
qcirc.append(iqft, count)
qcirc.barrier()

# Measure counting qubits
qcirc.measure(count, range(len(count)))

# print()
# print(qcirc.draw())

#=====================================================================================================================

emulator = Aer.get_backend('qasm_simulator')
job = execute(qcirc, emulator, shots=2048 )
hist = job.result().get_counts()
print(hist)

measured_int = int(max(hist, key=hist.get),2)
theta = (measured_int/(2**len(count)))*pi*2
counter = 2**len(search) * (1 - sin(theta/2)**2)
print("Number of solutions = %.1f" % counter)
		
#=====================================================================================================================