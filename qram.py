import numpy as np
import random
from qiskit import QuantumCircuit, Aer, execute
from math import log2, ceil, pi
from numpy import savetxt, save, savez_compressed

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

qnos = [dsz, tlog, tdim, hsz, csz, csz, tsz, nanc]

fsm     = list(range(sum(qnos[0:0]),sum(qnos[0:1])))
state   = list(range(sum(qnos[0:1]),sum(qnos[0:2])))  # States (Binary coded)
move    = list(range(sum(qnos[0:2]),sum(qnos[0:3])))
head    = list(range(sum(qnos[0:3]),sum(qnos[0:4])))  # Binary coded, 0-MSB 2-LSB, [001] refers to Tape pos 1, not 4
read    = list(range(sum(qnos[0:4]),sum(qnos[0:5])))
write   = list(range(sum(qnos[0:5]),sum(qnos[0:6])))  # Can be MUXed with read?
tape    = list(range(sum(qnos[0:6]),sum(qnos[0:7])))
ancilla = list(range(sum(qnos[0:7]),sum(qnos[0:8])))

# print("\nFSM\t:",fsm,"\nSTATE\t:",state,"\nMOVE\t:",move,"\nHEAD\t:",head,"\nREAD\t:",read,"\nWRITE\t:",write,"\nTAPE\t:",tape,"\nANCILLA :",ancilla)

#=====================================================================================================================

test 	= []
unit	= 'none'

qcirc_width = sum(qnos[0:8]) + len(test)
qcirc = QuantumCircuit(qcirc_width)

U_init(qcirc, qcirc_width, fsm)
# for tick in range(0, sim_tick):
# 	U_read(qcirc, read, head, tape, ancilla)
# 	U_fsm(qcirc, tick, fsm, state, read, write, move, ancilla)
# 	U_write(qcirc, write, head, tape, ancilla)
# 	U_move(qcirc, move, head, ancilla)
# 	U_rst(qcirc, tick, fsm, state, read, write, move, ancilla)

# print()
# print(qcirc.draw())
# print()
# print(qcirc.qasm())
print()
disp_isv(qcirc, "Step: Test all", all=False, precision=1e-4)

#=====================================================================================================================

def U_count(reg, val):
	if (len(reg) != len(val)):
		print("Search length not same")
	else:
		print("Search in progress....")
		
#=====================================================================================================================

U_count(fsm, val = '0000')

#=====================================================================================================================