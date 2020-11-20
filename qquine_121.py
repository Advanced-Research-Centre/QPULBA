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

# 24 qubits with Hadamard on 12 qubits log size: 880 MB csv, 816 MB txt, 256 MB npy, 255 KB npz
def save_isv(statevector, mode=1):
	if (mode == 1): savez_compressed('output.npz', statevector)
	elif (mode == 2): save('output.npy', statevector)
	elif (mode == 3):
		qb = int(log2(len(statevector)))
		f = open("output.txt", "w")	
		f.write("============ State Vector ============\n")
		s = 0
		for i in statevector:
			f.write('  ({:+.5f})   |{:0{}b}>'.format(i,s,qb)+'\n')
			s = s+1
		f.write("============..............============")
		f.close()		
	elif (mode == 4): savetxt('output.csv', statevector, delimiter=',')
	else: print('Invalid mode selected')
	return

#=====================================================================================================================

def nCX(k,c,t,b):
    nc = len(c)
    if nc == 1:
        k.cx(c[0], t[0])
    elif nc == 2:
        k.toffoli(c[0], c[1], t[0])
    else:
        nch = ceil(nc/2)
        c1 = c[:nch]
        c2 = c[nch:]
        c2.append(b[0])
        nCX(k,c1,b,[nch+1])
        nCX(k,c2,t,[nch-1])
        nCX(k,c1,b,[nch+1])
        nCX(k,c2,t,[nch-1])
    return

#=====================================================================================================================

def U_init(qcirc, circ_width, fsm):
	for i in fsm:                  
		qcirc.h(i)
	qcirc.barrier()
	return

def U_read(qcirc, read, head, tape, ancilla):
	# Reset read (prepz measures superposed states... need to uncompute)
	for cell in range(0, len(tape)):
		enc = format(cell, '#0'+str(len(head)+2)+'b')   # 2 for '0b' prefix
		for i in range(2, len(enc)):
			if(enc[i] == '0'):
				qcirc.x(head[(len(head)-1)-(i-2)])
		qcirc.barrier(read, head)
		nCX(qcirc, head+[tape[cell]], read, [ancilla[0]])
		qcirc.barrier(read, head)
		for i in range(2, len(enc)):
			if(enc[i] == '0'):
				qcirc.x(head[(len(head)-1)-(i-2)])
		qcirc.barrier(read, head, tape, ancilla)
	qcirc.barrier()
	return

def U_fsm(qcirc, tick, fsm, state, read, write, move, ancilla):
	# Description Number Encoding: {M/W}{R}
	# [ M1 W1 M0 W0 ] LSQ = W0 = fsm[0]
    qcirc.x(read[0])                                 	# If read == 0
    nCX(qcirc, [fsm[0],read[0]], write, [ancilla[0]])		# Update write
    nCX(qcirc, [fsm[1],read[0]], move, [ancilla[0]]) 		# Update move
    qcirc.x(read[0])                                 	# If read == 1
    nCX(qcirc, [fsm[2],read[0]], write, [ancilla[0]])		# Update write
    nCX(qcirc, [fsm[3],read[0]], move, [ancilla[0]]) 		# Update move
    qcirc.barrier()
    return

def U_write(qcirc, write, head, tape, ancilla):
    # Reset write (prepz measures superposed states... need to uncompute)
    for cell in range(0, len(tape)):
        enc = format(cell, '#0'+str(len(head)+2)+'b')   # 2 for '0b' prefix
        for i in range(2, len(enc)):
            if(enc[i] == '0'):
                qcirc.x(head[(len(head)-1)-(i-2)])
        qcirc.barrier(write, head)
        nCX(qcirc, head+write, [tape[cell]], [ancilla[0]])
        qcirc.barrier(write, head)
        for i in range(2, len(enc)):
            if(enc[i] == '0'):
                qcirc.x(head[(len(head)-1)-(i-2)])
        qcirc.barrier(write, head, tape, ancilla)		
    qcirc.barrier()
    return

def U_move(qcirc, move, head, ancilla):
	# Increment/Decrement using Adder

	reg_a = move
	reg_a.extend([-1]*(len(head)-len(move)))

	reg_b = head

	reg_c = [-1]        # No initial carry
	reg_c.extend(ancilla)
	reg_c.append(-1)    # Ignore Head position under/overflow. Trim bits. Last carry not accounted, All-ones overflows to All-zeros

	def q_carry(qcirc, q0, q1, q2, q3):
		if (q1 != -1 and q2 != -1 and q3 != -1):    qcirc.toffoli(q1, q2, q3)
		if (q1 != -1 and q2 != -1):                 qcirc.cx(q1, q2)
		if (q0 != -1 and q2 != -1 and q3 != -1):    qcirc.toffoli(q0, q2, q3)
	def q_mid(qcirc, q0, q1):
		if (q0 != -1 and q1 != -1):                 qcirc.cx(q0, q1)
	def q_sum(qcirc, q0, q1, q2):
		if (q0 != -1 and q2 != -1):                 qcirc.cx(q0, q2)
		if (q1 != -1 and q2 != -1):                 qcirc.cx(q1, q2)
	def q_rcarry(qcirc, q0, q1, q2, q3):
		if (q0 != -1 and q2 != -1 and q3 != -1):    qcirc.toffoli(q0, q2, q3)
		if (q1 != -1 and q2 != -1):                 qcirc.cx(q1, q2)
		if (q1 != -1 and q2 != -1 and q3 != -1):    qcirc.toffoli(q1, q2, q3)

	# Quantum Adder
	for i in range(0,len(head)):
		q_carry(qcirc,reg_c[i],reg_a[i],reg_b[i],reg_c[i+1])
	q_mid(qcirc,reg_a[i],reg_b[i])
	q_sum(qcirc,reg_c[i],reg_a[i],reg_b[i])
	for i in range(len(head)-2,-1,-1):
		q_rcarry(qcirc,reg_c[i],reg_a[i],reg_b[i],reg_c[i+1])
		q_sum(qcirc,reg_c[i],reg_a[i],reg_b[i])

	qcirc.x(reg_a[0])
	# Quantum Subtractor
	for i in range(0,len(head)-1):
		q_sum(qcirc,reg_c[i],reg_a[i],reg_b[i])
		q_carry(qcirc,reg_c[i],reg_a[i],reg_b[i],reg_c[i+1])
	q_sum(qcirc,reg_c[i+1],reg_a[i+1],reg_b[i+1])
	q_mid(qcirc,reg_a[i+1],reg_b[i+1])
	for i in range(len(head)-2,-1,-1):
		q_rcarry(qcirc,reg_c[i],reg_a[i],reg_b[i],reg_c[i+1])
	qcirc.x(reg_a[0])	
	
	qcirc.barrier()
	return

def U_rst(qcirc, tick, fsm, state, read, write, move, ancilla):
	# Reset write and move                                          
	qcirc.x(read[0])   
	nCX(qcirc, [fsm[0],read[0]], write, [ancilla[0]])      
	nCX(qcirc, [fsm[1],read[0]], move, [ancilla[0]])  
	qcirc.x(read[0])                                         
	nCX(qcirc, [fsm[2],read[0]], write, [ancilla[0]])     
	nCX(qcirc, [fsm[3],read[0]], move, [ancilla[0]])		
	qcirc.barrier()    
	return

#=====================================================================================================================

def Test_cfg_121(block):	# convert config from 221 to 121
	global fsm, state, move, head, read, write, tape, ancilla, test
	if (block == 'none'):
		return
	elif (block == 'read'):
		fsm     = []
		state   = []
		move    = []
		head    = [0,1,2,3]
		read    = [4]
		write   = []
		tape    = [5,6,7,8,9,10,11,12,13,14,15,16]
		ancilla = [17]
		test 	= [18]
	elif (block == 'fsm'):
		fsm     = [0,1,2,3,4,5,6,7,8,9,10,11]
		state   = [12,13]
		move    = [14]
		head    = []
		read    = [15]
		write   = [16]
		tape    = []
		ancilla = [17]
		test 	= [18,19,20]
	elif (block == 'move'):
		fsm     = []
		state   = []
		move    = [0]
		head    = [1,2,3,4]
		read    = []
		write   = []
		tape    = []
		ancilla = [5,6,7]
		test 	= [8,9,10,11]
	elif (block == 'write'):
		fsm     = []
		state   = []
		move    = []
		head    = [0,1,2,3]
		read    = []
		write   = [4]
		tape    = [5,6,7,8,9,10,11,12,13,14,15,16]
		ancilla = [17]
		test 	= []#[18,19,20,21,22,23,24,25,26,27,28,29]
	elif (block == 'rst'):
		fsm     = [0,1,2,3,4,5,6,7,8,9,10,11]
		state   = [12,13]
		move    = [14]
		head    = []
		read    = [15]
		write   = [16]
		tape    = []
		ancilla = [17]
		test 	= [18,19,20,21]
	elif (block == 'count'):
		fsm     = [0,1,2,3]
		state   = []
		move    = []
		head    = []
		read    = []
		write   = []
		tape    = []
		ancilla = []
		test 	= []
		count	= [4,5,6,7]
		search 	= [8,9,10,11]
	print("\n\nTEST CONFIGURATION\n\tFSM\t:",fsm,"\n\tSTATE\t:",state,"\n\tMOVE\t:",move,"\n\tHEAD\t:",head,"\n\tREAD\t:",read,"\n\tWRITE\t:",write,"\n\tTAPE\t:",tape,"\n\tANCILLA :",ancilla,"\n\tTEST\t:",test,"\n\tCOUNT\t:",count,"\n\tSEARCH\t:",search)

def Test_count(qcirc, fsm):
	# Test using some superposition of fsm and then Hamming distance
	qcirc.barrier()
	qcirc.barrier()
	return

#=====================================================================================================================

asz = 2                                         # Alphabet size: Binary (0 is blank/default)
ssz = 1                                         # State size (Initial state is all 0)
tdim = 1                                        # Tape dimension

csz = ceil(log2(asz))                           # Character symbol size
senc = ceil(log2(ssz))                          # State encoding size
transitions = ssz * asz                         # Number of transition arrows in FSM
dsz = transitions * (tdim + csz + senc)         # Description size

machines = 2 ** dsz
print("\nNumber of "+str(asz)+"-symbol "+str(ssz)+"-state "+str(tdim)+"-dimension Quantum Parallel Universal Linear Bounded Automata: "+str(machines))

tsz = dsz                                       # Turing Tape size (same as dsz to estimating self-replication and algorithmic probability)
hsz = ceil(log2(tsz))                           # Head size

sim_tick = tsz                                  # Number of ticks of the FSM before abort
#sim_tick = 1                                  	# Just 1 QPULBA cycle for proof-of-concept
tlog = (sim_tick+1) * senc                      # Transition log # required?
nanc	= 3

countbits = 0

qnos = [dsz, tlog, tdim, hsz, csz, csz, tsz, nanc, countbits]

# searchbits = 5
# for j in range(1,8):
# 	print("\nDetectable solutions with %d count bits:",j)
# 	countbits = j
# 	for i in range(0,countbits**2):
# 		theta = (i/(2**countbits))*pi*2
# 		counter = 2**searchbits * (1 - sin(theta/2)**2)
# 		print(round(counter),"|",end='')
# sys.exit(0)

fsm     = list(range(sum(qnos[0:0]),sum(qnos[0:1])))
state   = list(range(sum(qnos[0:1]),sum(qnos[0:2])))  # States (Binary coded)
move    = list(range(sum(qnos[0:2]),sum(qnos[0:3])))
head    = list(range(sum(qnos[0:3]),sum(qnos[0:4])))  # Binary coded, 0-MSB 2-LSB, [001] refers to Tape pos 1, not 4
read    = list(range(sum(qnos[0:4]),sum(qnos[0:5])))
write   = list(range(sum(qnos[0:5]),sum(qnos[0:6])))  # Can be MUXed with read?
tape    = list(range(sum(qnos[0:6]),sum(qnos[0:7])))
ancilla = list(range(sum(qnos[0:7]),sum(qnos[0:8])))

count	= list(range(sum(qnos[0:8]),sum(qnos[0:9])))
# search	= list(range(sum(qnos[0:9]),sum(qnos[0:10])))

print("\nFSM\t:",fsm,"\nSTATE\t:",state,"\nMOVE\t:",move,"\nHEAD\t:",head,"\nREAD\t:",read,"\nWRITE\t:",write,"\nTAPE\t:",tape,"\nANCILLA :",ancilla,"\nCOUNT\t:",count)


#=====================================================================================================================

test 	= []
unit	= 'none'	# 'none', 'read', 'fsm', 'write', 'move', 'rst', 'count'

qcirc_width = sum(qnos[0:9]) + len(test)
if (countbits != 0):
	qcirc = QuantumCircuit(qcirc_width, len(count))
else:
	qcirc = QuantumCircuit(qcirc_width, len(tape))

# U_init(qcirc, qcirc_width, fsm)
# for tick in range(0, sim_tick):
# 	U_read(qcirc, read, head, tape, ancilla)
# 	U_fsm(qcirc, tick, fsm, state, read, write, move, ancilla)
# 	U_write(qcirc, write, head, tape, ancilla)
# 	U_move(qcirc, move, head, ancilla)
# 	U_rst(qcirc, tick, fsm, state, read, write, move, ancilla)

#	============ State Vector ============ Step: Run QPULBA 121
#	(+0.25000+0.00000j)   |0000000000000000>
#	(+0.25000+0.00000j)   |0000000000000100>
#	(+0.25000+0.00000j)   |0000000000001000>
#	(+0.25000+0.00000j)   |0000000000001100>
#	(+0.25000+0.00000j)   |0001111000000001>
#	(+0.25000+0.00000j)   |0001111000000101>
#	(+0.25000+0.00000j)   |0001111000001001>
#	(+0.25000+0.00000j)   |0001111000001101>
#	(+0.25000+0.00000j)   |0100000000000010>
#	(+0.25000+0.00000j)   |0100000000000110>
#	(+0.25000+0.00000j)   |0100000000001010>
#	(+0.25000+0.00000j)   |0100000000001110>
#	(+0.25000+0.00000j)   |0101111000000011>
#	(+0.25000+0.00000j)   |0101111000000111>
#	(+0.25000+0.00000j)   |0101111000001011>
#	(+0.25000+0.00000j)   |0101111000001111>
#	============..............============

def U_qpulba121(qcirc, fsm, tape, ancilla):
	qcirc.h(fsm)
	qcirc.cx(fsm[1], ancilla[1])
	qcirc.cx(fsm[0],tape[0])
	qcirc.cx(fsm[0],tape[1])
	qcirc.cx(fsm[0],tape[2])
	qcirc.cx(fsm[0],tape[3])
	# qcirc.h(tape[1])
	# qcirc.h(tape[2])
	# qcirc.h(tape[3])
	return

U_qpulba121(qcirc, fsm, tape, ancilla)

disp_isv(qcirc, "Step: Run QPULBA 121", all=False, precision=1e-4)

#=====================================================================================================================

def condition_fsm(qcirc, fsm, tape):
	# Finding specific programs-output characteristics			(fsm|tape)
	# e.g. Self-replication
	for q in fsm:
		qcirc.cx(q,tape[q])
	qcirc.barrier()
	return

def condition_tape(qcirc, tape, target_tape):
	# Finding algorithmic probability of a specific output		(tape|tape*)
	return

def condition_state(qcirc, state, target_state):
	# Finding programs with specific end state					(state|state*)
	# Note: not possible in QPULBA 121
	return

#=====================================================================================================================

search = tape
condition_fsm(qcirc, fsm, tape)

disp_isv(qcirc, "Step: Find self-replicating programs", all=False, precision=1e-4)

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

def U_pattern(sz):
	# Mark {0000,0010,0100,0110,1000,1010,1100,1110}
	tgt_reg = list(range(0,sz))
	oracle = QuantumCircuit(len(tgt_reg))
	oracle.x(tgt_reg)
	oracle.h(tgt_reg[0])
	oracle.mct(tgt_reg[1:],tgt_reg[0])
	oracle.h(tgt_reg[0])
	oracle.x(tgt_reg)
	oracle.x([tgt_reg[0],tgt_reg[2],tgt_reg[3]])
	oracle.h(tgt_reg[0])
	oracle.mct(tgt_reg[1:],tgt_reg[0])
	oracle.h(tgt_reg[0])
	oracle.x([tgt_reg[0],tgt_reg[2],tgt_reg[3]])
	oracle.x([tgt_reg[0],tgt_reg[1],tgt_reg[3]])
	oracle.h(tgt_reg[0])
	oracle.mct(tgt_reg[1:],tgt_reg[0])
	oracle.h(tgt_reg[0])
	oracle.x([tgt_reg[0],tgt_reg[1],tgt_reg[3]])
	oracle.x([tgt_reg[0],tgt_reg[3]])
	oracle.h(tgt_reg[0])
	oracle.mct(tgt_reg[1:],tgt_reg[0])
	oracle.h(tgt_reg[0])
	oracle.x([tgt_reg[0],tgt_reg[3]])
	oracle.x([tgt_reg[0],tgt_reg[1],tgt_reg[2]])
	oracle.h(tgt_reg[0])
	oracle.mct(tgt_reg[1:],tgt_reg[0])
	oracle.h(tgt_reg[0])
	oracle.x([tgt_reg[0],tgt_reg[1],tgt_reg[2]])
	oracle.x([tgt_reg[0],tgt_reg[2]])
	oracle.h(tgt_reg[0])
	oracle.mct(tgt_reg[1:],tgt_reg[0])
	oracle.h(tgt_reg[0])
	oracle.x([tgt_reg[0],tgt_reg[2]])
	oracle.x([tgt_reg[0],tgt_reg[1]])
	oracle.h(tgt_reg[0])
	oracle.mct(tgt_reg[1:],tgt_reg[0])
	oracle.h(tgt_reg[0])
	oracle.x([tgt_reg[0],tgt_reg[1]])
	oracle.x([tgt_reg[0]])
	oracle.h(tgt_reg[0])
	oracle.mct(tgt_reg[1:],tgt_reg[0])
	oracle.h(tgt_reg[0])
	oracle.x([tgt_reg[0]])
	return oracle

def U_aa(sz):
	tgt_reg = list(range(0,sz))
	diffuser = QuantumCircuit(len(tgt_reg))
	diffuser.h(tgt_reg)
	diffuser.x(tgt_reg)
	diffuser.h(tgt_reg[0])
	diffuser.mct(tgt_reg[1:],tgt_reg[0])
	diffuser.h(tgt_reg[0])
	diffuser.x(tgt_reg)
	diffuser.h(tgt_reg)
	return diffuser

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

oracle = U_oracle(len(tape)).to_gate()
oracle.label = "GO"

pattern = U_pattern(len(tape)).to_gate()
pattern.label = "PO"

allregs = list(range(sum(qnos[0:0]),sum(qnos[0:8])))
aa = U_aa(len(tape)).to_gate()
aa.label = "AA"

from copy import deepcopy

def count_constructors(qcirc, gi):

	for i in range(gi):
		qcirc.append(oracle, tape)
		disp_isv(qcirc, "Step: Mark", all=False, precision=1e-4)
		qcirc.append(aa, tape)
		# disp_isv(qcirc, "Step: Amplify", all=False, precision=1e-4)

	qcirc.measure(tape, range(len(tape)))
		
	emulator = Aer.get_backend('qasm_simulator')
	job = execute(qcirc, emulator, shots=2048)
	hist = job.result().get_counts()
	print(hist)
	return

# def count_constructors(qcirc, gi):

# 	qcirc.append(oracle, tape)
# 	qcirc.append(aa, tape)
# 	qcirc.append(pattern, tape)
# 	qcirc.append(aa, tape)
# 	for i in range(gi):
# 		qcirc.append(oracle, tape)
# 		disp_isv(qcirc, "Step: Mark", all=False, precision=1e-4)
# 		qcirc.append(aa, tape)
# 		# disp_isv(qcirc, "Step: Amplify", all=False, precision=1e-4)

# 	qcirc.measure(tape, range(len(tape)))
	
# 	# print()
# 	# print(qcirc.draw())
		
# 	emulator = Aer.get_backend('qasm_simulator')
# 	job = execute(qcirc, emulator, shots=2048)
# 	hist = job.result().get_counts()
# 	print(hist)
# 	return

for i in range(0,3):
	count_constructors(deepcopy(qcirc),i)

























#=====================================================================================================================
# Code below archived for now	
#=====================================================================================================================

def count_constructors():


	#=====================================================================================================================

	# Create controlled Grover oracle circuit
	oracle = U_oracle(len(search)).to_gate()
	c_oracle = oracle.control()
	c_oracle.label = "cGO"

	# Create controlled Grover diffuser circuit
	# diffuser = U_diffuser(len(search)).to_gate()
	allregs = list(range(sum(qnos[0:0]),sum(qnos[0:8])))
	# selregs = [0,1,2,3,9,10,11,12,14]	# fsm, tape, ancilla[1]
	selregs = [0,9,10,11,12]	# fsm[0], tape
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
			qcirc.append(c_oracle, [qb] + search)
			qcirc.append(c_diffuser, [qb] + selregs)
		iterations *= 2
		qcirc.barrier()

	# Inverse QFT
	qcirc.append(iqft, count)
	qcirc.barrier()

	# print()
	# disp_isv(qcirc, "Step: Search and count", all=False, precision=1e-4)

	# Measure counting qubits
	qcirc.measure(count, range(len(count)))

	# print()
	# print(qcirc.draw())

	# sys.exit(0)

	#=====================================================================================================================

	emulator = Aer.get_backend('qasm_simulator')
	job = execute(qcirc, emulator, shots=128)
	hist = job.result().get_counts()
	# print(hist)

	measured_int = int(max(hist, key=hist.get),2)
	theta = (measured_int/(2**len(count)))*pi*2
	counter = 2**len(selregs) * (1 - sin(theta/2)**2)
	print("Number of solutions = %.1f" % counter)
			
	#=====================================================================================================================