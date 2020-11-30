import numpy as np
import random
from qiskit import QuantumCircuit, Aer, execute
from math import log2
import sys

#=====================================================================================================================

simulator = Aer.get_backend('statevector_simulator')
emulator = Aer.get_backend('qasm_simulator')

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

qcirc = QuantumCircuit(3,1)

qcirc.h(0)
qcirc.h(1)

disp_isv(qcirc, "Step 1", all=False, precision=1e-4)

qcirc.ccx(0,1,2)

disp_isv(qcirc, "Step 2", all=False, precision=1e-4)

qcirc.measure(2, 0)

print(qcirc.draw())

for i in range(0,4):
	disp_isv(qcirc, "Step 3 Try "+str(i+1), all=False, precision=1e-4)

# job = execute(qcirc, emulator, shots=1024)
# hist = job.result().get_counts()
		
#=====================================================================================================================

#		(qeait) D:\GoogleDrive\RESEARCH\0 - Programs\QPULBA>python qpostselect.py
#
#		============ State Vector ============ Step 1
#		  (+0.50000+0.00000j)   |000>
#		  (+0.50000+0.00000j)   |001>
#		  (+0.50000+0.00000j)   |010>
#		  (+0.50000+0.00000j)   |011>
#		============..............============
#
#		============ State Vector ============ Step 2
#		  (+0.50000+0.00000j)   |000>
#		  (+0.50000+0.00000j)   |001>
#		  (+0.50000+0.00000j)   |010>
#		  (+0.50000+0.00000j)   |111>
#		============..............============
#			 ┌───┐
#		q_0: ┤ H ├──■─────
#			 ├───┤  │
#		q_1: ┤ H ├──■─────
#			 └───┘┌─┴─┐┌─┐
#		q_2: ─────┤ X ├┤M├
#				  └───┘└╥┘
#		c: 1/═══════════╩═
#						0
#
#		============ State Vector ============ Step 3 Try 1
#		  (+0.57735+0.00000j)   |000>
#		  (+0.57735+0.00000j)   |001>
#		  (+0.57735+0.00000j)   |010>
#		============..............============
#
#		============ State Vector ============ Step 3 Try 2
#		  (+1.00000+0.00000j)   |111>
#		============..............============
#
#		============ State Vector ============ Step 3 Try 3
#		  (+0.57735+0.00000j)   |000>
#		  (+0.57735+0.00000j)   |001>
#		  (+0.57735+0.00000j)   |010>
#		============..............============
#
#		============ State Vector ============ Step 3 Try 4
#		  (+0.57735+0.00000j)   |000>
#		  (+0.57735+0.00000j)   |001>
#		  (+0.57735+0.00000j)   |010>
#		============..............============

#=====================================================================================================================