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
	return

def U_read(qcirc, read, head, tape, ancilla):
	# Reset read (prepz measures superposed states... need to uncompute)
	for cell in range(0, len(tape)):
		enc = format(cell, '#0'+str(len(head)+2)+'b')   # 2 for '0b' prefix
		for i in range(2, len(enc)):
			if(enc[i] == '0'):
				qcirc.x(head[(len(head)-1)-(i-2)])
		nCX(qcirc, head+[tape[cell]], read, [ancilla[0]])
		for i in range(2, len(enc)):
			if(enc[i] == '0'):
				qcirc.x(head[(len(head)-1)-(i-2)])
		qcirc.barrier()
	return

def U_fsm(qcirc, tick, fsm, state, read, write, move, ancilla):
	# Description Number Encoding: {Q/M/W}{QR}
	# [ Q11 M11 W11 Q10 M10 W10 Q01 M01 W01 Q00 M00 W00 ] LSQ = W00 = fsm[0]
    qcirc.x(state[tick])                                             			# If s == 0
    qcirc.x(read[0])                                                  				# If s == 0 && read == 0
    nCX(qcirc, [state[tick],fsm[0],read[0]], write, [ancilla[0]])                 		# Update write
    nCX(qcirc, [state[tick],fsm[1],read[0]], move, [ancilla[0]])                  		# Update move
    nCX(qcirc, [state[tick],fsm[2],read[0]], [state[tick+1]], [ancilla[0]])       		# Update state
    qcirc.x(read[0])                                                 				# If s == 0 && read == 1
    nCX(qcirc, [state[tick],fsm[3],read[0]], write, [ancilla[0]])                 		# Update write
    nCX(qcirc, [state[tick],fsm[4],read[0]], move, [ancilla[0]])                  		# Update move
    nCX(qcirc, [state[tick],fsm[5],read[0]], [state[tick+1]], [ancilla[0]])       		# Update state
    qcirc.x(state[tick])	                                             		# If s == 1
    qcirc.x(read[0])                                                  				# If s == 1 && read == 0
    nCX(qcirc, [state[tick],fsm[6],read[0]], write, [ancilla[0]])                 		# Update write
    nCX(qcirc, [state[tick],fsm[7],read[0]], move, [ancilla[0]])                  		# Update move
    nCX(qcirc, [state[tick],fsm[8],read[0]], [state[tick+1]], [ancilla[0]])       		# Update state
    qcirc.x(read[0])                                                 				# If s == 1 && read == 1
    nCX(qcirc, [state[tick],fsm[9],read[0]], write, [ancilla[0]])                 		# Update write
    nCX(qcirc, [state[tick],fsm[10],read[0]], move, [ancilla[0]])                  		# Update move
    nCX(qcirc, [state[tick],fsm[11],read[0]], [state[tick+1]], [ancilla[0]])       		# Update state
    return

def U_write(qcirc, write, head, tape, ancilla):
    # Reset write (prepz measures superposed states... need to uncompute)
    for cell in range(0, len(tape)):
        enc = format(cell, '#0'+str(len(head)+2)+'b')   # 2 for '0b' prefix
        for i in range(2, len(enc)):
            if(enc[i] == '0'):
                qcirc.x(head[(len(head)-1)-(i-2)])
        nCX(qcirc, head+write, [tape[cell]], [ancilla[0]])
        for i in range(2, len(enc)):
            if(enc[i] == '0'):
                qcirc.x(head[(len(head)-1)-(i-2)])
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
	
	return

def U_rst(qcirc, tick, fsm, state, read, write, move, ancilla):
	# Reset write and move
	qcirc.x(state[tick])                                           
	qcirc.x(read[0])   
	nCX(qcirc, [state[tick],fsm[0],read[0]], write, [ancilla[0]])      
	nCX(qcirc, [state[tick],fsm[1],read[0]], move, [ancilla[0]])  
	qcirc.x(read[0])                                         
	nCX(qcirc, [state[tick],fsm[3],read[0]], write, [ancilla[0]])     
	nCX(qcirc, [state[tick],fsm[4],read[0]], move, [ancilla[0]])    
	qcirc.x(state[tick])                                        
	qcirc.x(read[0])                                                 
	nCX(qcirc, [state[tick],fsm[6],read[0]], write, [ancilla[0]])      
	nCX(qcirc, [state[tick],fsm[7],read[0]], move, [ancilla[0]])  
	qcirc.x(read[0])                                         
	nCX(qcirc, [state[tick],fsm[9],read[0]], write, [ancilla[0]])     
	nCX(qcirc, [state[tick],fsm[10],read[0]], move, [ancilla[0]])    
	# Maintain computation history
	qcirc.swap(state[0],state[tick+1])
	return

#=====================================================================================================================

def Test_cfg(block):
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
	print("\n\nTEST CONFIGURATION\n\tFSM\t:",fsm,"\n\tSTATE\t:",state,"\n\tMOVE\t:",move,"\n\tHEAD\t:",head,"\n\tREAD\t:",read,"\n\tWRITE\t:",write,"\n\tTAPE\t:",tape,"\n\tANCILLA :",ancilla,"\n\tTEST\t:",test)

def Test_read(qcirc, read, head, tape, ancilla, test):
	# Test using full superposition of head and some random tape qubits
	# Test associated to read
	for i in range(0,len(head)):
		qcirc.h(head[i])
	# Create random binary string of length tape 
	randbin = ""
	for i in range(len(tape)): randbin += str(random.randint(0, 1)) 
	for i in range(0,len(tape)):
		if (randbin[i] == '1'):
			qcirc.h(tape[i])	# Replace H with X for ease
	qcirc.cx(read[0],test[0])
	print("Test tape:",randbin) 
	qcirc.barrier()
	return

def Test_fsm(qcirc, tick, fsm, state, read, write, move, ancilla, test):
	# Test using full superposition of fsm, current state, read
	# Test associated to move, write, new state
	# fsm superposition part of U_init
	qcirc.barrier()
	qcirc.h(state[0])
	qcirc.h(read[0])
	qcirc.barrier()
	qcirc.cx(write[0],test[0])
	qcirc.cx(move[0],test[1])
	qcirc.cx(state[1],test[2])
	qcirc.barrier()
	return

def Test_write(qcirc, write, head, tape, ancilla, test):
	# Test using full superposition of head and write
	# Test associated to tape (optional)
	for i in range(0,len(head)):
		qcirc.h(head[i])
	qcirc.h(write)
	# for i in range(0,len(tape)):
	# 	qcirc.cx(tape[i],test[i])
	return

def Test_move(qcirc, move, head, ancilla, test):
	# Test using full superposition of head, both inc/dec
	# Test associated to head
	for i in range(0,len(head)):
		qcirc.h(head[i])
		qcirc.cx(head[i],test[i]) 
	qcirc.h(move[0])
	qcirc.barrier()
	return

def Test_rst(qcirc, tick, fsm, state, read, write, move, ancilla, test):
	# Test using full superposition of fsm, current state, read
	# Test associated to move, write, new state
	# fsm superposition part of U_init
	for i in range(0,len(state)):
		qcirc.h(state[i])
	qcirc.h(read[0])
	qcirc.h(write[0])
	qcirc.h(move[0])
	qcirc.barrier()
	for i in range(0,len(state)):
		qcirc.cx(state[i],test[i])
	qcirc.cx(write[0],test[len(state)])
	qcirc.cx(move[0],test[len(state)+1])
	qcirc.barrier()
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
print("\nNumber of "+str(asz)+"-symbol "+str(ssz)+"-state "+str(tdim)+"-dimension Quantum Parallel Universal Linear Bounded Automata: "+str(machines))

tsz = dsz                                       # Turing Tape size (same as dsz to estimating self-replication and algorithmic probability)
hsz = ceil(log2(tsz))                           # Head size

sim_tick = tsz                                  # Number of ticks of the FSM before abort
sim_tick = 1                                  	# Just 1 QPULBA cycle for proof-of-concept
tlog = (sim_tick+1) * senc                      # Transition log # required?
nanc	= 3

fsm     = list(range(   0,              dsz                         ))
state   = list(range(   fsm     [-1]+1, fsm     [-1]+1+     tlog    ))  # States (Binary coded)
move    = list(range(   state   [-1]+1, state   [-1]+1+     tdim    ))
head    = list(range(   move    [-1]+1, move    [-1]+1+     hsz     ))  # Binary coded, 0-MSB 2-LSB, [001] refers to Tape pos 1, not 4
read    = list(range(   head    [-1]+1, head    [-1]+1+     csz     ))
write   = list(range(   read    [-1]+1, read    [-1]+1+     csz     ))  # Can be MUXed with read?
tape    = list(range(   write   [-1]+1, write   [-1]+1+     tsz     ))
ancilla = list(range(   tape    [-1]+1, tape    [-1]+1+     nanc    ))
print("\nFSM\t:",fsm,"\nSTATE\t:",state,"\nMOVE\t:",move,"\nHEAD\t:",head,"\nREAD\t:",read,"\nWRITE\t:",write,"\nTAPE\t:",tape,"\nANCILLA :",ancilla)

#=====================================================================================================================

test 	= []
unit	= 'none'	# 'read', 'fsm', 'write', 'move', 'rst'
Test_cfg(unit)

qcirc_width = len(fsm) + len(state) + len(move) + len(head) + len(read) + len(write) + len(tape) + len(ancilla) + len(test)
qcirc = QuantumCircuit(qcirc_width)

# 1. Initialize
U_init(qcirc, qcirc_width, fsm)

# 2. Run machine for n-iterations:
for tick in range(0, sim_tick):
	
	# 2.1	{read} << U_read({head, tape})
	if (unit == 'read'): Test_read(qcirc, read, head, tape, ancilla, test)
	U_read(qcirc, read, head, tape, ancilla)

	# 2.2	{write, state, move} << U_fsm({read, state, fsm})
	if (unit == 'fsm'): Test_fsm(qcirc, tick, fsm, state, read, write, move, ancilla, test)
	U_fsm(qcirc, tick, fsm, state, read, write, move, ancilla)
	
	# 2.3	{tape} << U_write({head, write})
	if (unit == 'write'): Test_write(qcirc, write, head, tape, ancilla, test)
	U_write(qcirc, write, head, tape, ancilla)

	# 2.4	{head, err} << U_move({head, move})
	if (unit == 'move'): Test_move(qcirc, move, head, ancilla, test)
	U_move(qcirc, move, head, ancilla)

	# 2.5  	reset
	if (unit == 'rst'): Test_rst(qcirc, tick, fsm, state, read, write, move, ancilla, test)
	U_rst(qcirc, tick, fsm, state, read, write, move, ancilla)

print()
#print(qcirc.draw())
print(qcirc.qasm())
# disp_isv(qcirc, "Step: Test all", all=False, precision=1e-4) # Full simulation doesn't work

#=====================================================================================================================