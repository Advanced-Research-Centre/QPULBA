from qiskit import QuantumCircuit
import numpy as np
import random
from qiskit import Aer, execute
from math import log2, ceil, pi

circ = QuantumCircuit(8)

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
			if (abs(i) > precision): print('  ({:+.5f})   |{:0{}b}>'.format(i,s,qb))
		s = s+1
	print("============..............============")
	return

# Step 1
circ.h(1)#ry(round(pi * random.random(),2),1)
circ.h(2)#ry(round(pi * random.random(),2),2)
circ.barrier()
disp_isv(circ,"Step 1",False,1e-4)

# Step 2
aU0 = round(pi * random.random(),2)
circ.ry(aU0,0)
circ.barrier()
disp_isv(circ,"Step 2",False,1e-4)

# Step 3
#circ.cx(0,4)
#circ.cx(1,5)
#circ.cx(2,6)
circ.barrier()
disp_isv(circ,"Step 3",False,1e-4)

# Step 4
circ.x(0)
circ.toffoli(0,1,3)
circ.x(0)
circ.toffoli(0,2,3)
circ.barrier()
disp_isv(circ,"Step 4",False,1e-4)

# Step 5
circ.cx(3,7)
circ.barrier()
disp_isv(circ,"Step 4",False,1e-4)

# Step 4
circ.toffoli(0,2,3)
circ.x(0)
circ.toffoli(0,1,3)
circ.x(0)
circ.barrier()
disp_isv(circ,"Step 4",False,1e-4)

## Step 5
#circ.ry(-aU0,0)
#circ.h(1)
#circ.h(2)
#circ.barrier()
#disp_isv(circ,"Step 5",False,1e-4)

## Step 6
#circ.swap(0,3)
#disp_isv(circ,"Step 6",False,1e-4)

print()
print(circ.draw())