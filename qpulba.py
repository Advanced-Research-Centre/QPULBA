import numpy as np
import random
from qiskit import QuantumCircuit, Aer, execute
from math import log2, ceil, pi
from numpy import savetxt, save, savez_compressed


# a2 = np.pi * random.random()
#import qsdk     # from qsdk import nCX

#=====================================================================================================================

simulator = Aer.get_backend('statevector_simulator')

def disp_isv(circ, msg="", all=True, precision=1e-8):
	sim_res = execute(circ, simulator).result()
	statevector = sim_res.get_statevector(circ)
	qb = int(log2(len(statevector)))
	print("\n============ State Vector ============", msg)
	s = 0
	for i in statevector:
		if (all == True): print('  ({:.5f})   |{:0{}b}>'.format(i,s,qb))
		else:
			if (abs(i) > precision): print('  ({:.5f})   |{:0{}b}>'.format(i,s,qb))
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
			f.write('  ({:.5f})   |{:0{}b}>'.format(i,s,qb)+'\n')
			s = s+1
		f.write("============..............============")
		f.close()		
	elif (mode == 4): savetxt('output.csv', statevector, delimiter=',')
	else: print('Invalid mode selected')
	return

#=====================================================================================================================

def U_init(qcirc, circ_width, fsm):
	for i in fsm:                  
		qcirc.h(i)
	return

def U_move(qcirc, move, head, ancilla, test):
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
	
	return

def Test_move(qcirc, move, head, ancilla, test):
	# Test using full superposition of head, both inc/dec and association to initial state
	for i in range(0,len(head)):
		qcirc.h(head[i])
		qcirc.cx(head[i],test[i]) 
	qcirc.h(move[0])
	return

#=====================================================================================================================

asz = 2                                         # Alphabet size: Binary (0 is blank/default)
ssz = 2                                         # State size (Initial state is all 0)
tdim = 1                                        # Tape dimension

csz = ceil(log2(asz))                           # Character symbol size
senc = ceil(log2(ssz))                          # State encoding size
transitions = ssz * asz                         # Number of transition arrows in FSM
dsz = transitions * (tdim + csz + senc)         # Description size

machines = 2 ** dsz
print("\nNumber of "+str(asz)+"-symbol "+str(ssz)+"-state"+str(tdim)+"-dimension Quantum Parallel Universal Linear Bounded Automata: "+str(machines))

tsz = dsz                                       # Turing Tape size (same as dsz to estimating self-replication and algorithmic probability)
hsz = ceil(log2(tsz))                           # Head size

sim_tick = tsz                                  # Number of ticks of the FSM before abort
tlog = (sim_tick+1) * senc                      # Transition log # required?
nanc	= 1

fsm     = list(range(   0,              dsz                         ))
state   = list(range(   fsm     [-1]+1, fsm     [-1]+1+     senc    ))  # States (Binary coded) # tlog?
move    = list(range(   state   [-1]+1, state   [-1]+1+     tdim    ))
head    = list(range(   move    [-1]+1, move    [-1]+1+     hsz     ))  # Binary coded, 0-MSB 2-LSB, [001] refers to Tape pos 1, not 4
read    = list(range(   head    [-1]+1, head    [-1]+1+     csz     ))
write   = list(range(   read    [-1]+1, read    [-1]+1+     csz     ))  # Can be MUXed with read?
tape    = list(range(   write   [-1]+1, write   [-1]+1+     tsz     ))
ancilla = list(range(   tape    [-1]+1, tape    [-1]+1+     nanc    ))
print("\nFSM\t:",fsm,"\nSTATE\t:",state,"\nMOVE\t:",move,"\nHEAD\t:",head,"\nREAD\t:",read,"\nWRITE\t:",write,"\nTAPE\t:",tape,"\nANCILLA :",ancilla)

# Test configuration
fsm     = []
state   = []
move    = [0]
head    = [1,2,3,4]
read    = []
write   = []
tape    = []
ancilla = [5,6,7]
test 	= [8,9,10,11]
print("\n\nTEST CONFIGURATION\n\tFSM\t:",fsm,"\n\tSTATE\t:",state,"\n\tMOVE\t:",move,"\n\tHEAD\t:",head,"\n\tREAD\t:",read,"\n\tWRITE\t:",write,"\n\tTAPE\t:",tape,"\n\tANCILLA :",ancilla,"\n\tTEST\t:",test)

qcirc_width = len(fsm) + len(state) + len(move) + len(head) + len(read) + len(write) + len(tape) + len(ancilla) + len(test)

#=====================================================================================================================

qcirc = QuantumCircuit(qcirc_width)

# 1. Initialize
U_init(qcirc, qcirc_width, fsm)

# 2. Run machine for n-iterations:
for tick in range(0, sim_tick):

	# 2.1  {read} << U_read({head, tape})
	# U_read(k_read, read, head, tape, move)   # move qubits used as borrowed ancilla

	# 2.2  {write, state, move} << U_fsm({read, state, fsm})
	# U_fsm(k_fsm, tick, fsm, state, read, write, move, ancilla)

	# 2.3  {tape} << U_write({head, write})
	# U_write(k_write, write, head, tape, ancilla)

	# 2.4  {head, err} << U_move({head, move})
	Test_move(qcirc, move, head, ancilla, test)
	U_move(qcirc, move, head, ancilla, test)

	# 2.5  UNCOMPUTE
	# U_fsm_UC(k_fsm_uc, tick, fsm, state, read, write, move, ancilla)  # TBD: Generalize tick = 0, inside sim_tick loop

	break

print(qcirc.draw())
disp_isv(qcirc, "Step: Test move", all=False, precision=1e-4)

#=====================================================================================================================