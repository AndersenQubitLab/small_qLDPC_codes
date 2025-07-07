import htlogicalgates as htlg
import numpy as np
import sys
import io

# X and Z logical operator for distance 3 LDPC code

x_logicals_ldpc = ["X0 X1 X3 X4 X6 X9", "X0 X1 X2"]
z_logicals_ldpc = ["Z0 Z2 Z4 Z5 Z6","Z0 Z1 Z2 Z3 Z4 Z5"]

# minimal stabilizer set

stabilizers_ldpc = ["X0 X5 X9 X10",
                    "X1 X5 X6 X8",
                    "X2 X5 X7 X8 X10 X11",
                    "X3 X5 X6 X8 X10 X11",
                    "X4 X5 X7 X8 X9 X10",
                    'Z0 Z2 Z7 Z9',
                    'Z1 Z2 Z7 Z8 Z9 Z10',
                    'Z3 Z5 Z7 Z8 Z9 Z11',
                    'Z4 Z5 Z8 Z9',
                    'Z6 Z7 Z8 Z9 Z10 Z11']


# define the code

stab_code_ldpc = htlg.StabilizerCode(x_logicals_ldpc, z_logicals_ldpc, stabilizers_ldpc)


# define the adjacency matrix

adjacency_matrix_ldpc = np.array([
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0]  
])

# define connectivity

#connectivity_ldpc = htlg.Connectivity(adjacency_matrix_ldpc)
connectivity_ldpc = htlg.Connectivity("all", num_qubits=12)


# define the circuit you want to find
# logical gate on two qubits
logical_gate_ldpc = htlg.Circuit(2)
# CNOT
#logical_gate_ldpc.cx(0,1)
#hadamard
logical_gate_ldpc.h(0)

# run the optimization

# circ, status = htlg.tailor_logical_gate(stab_code_ldpc, connectivity_ldpc,logical_gate_ldpc, num_cz_layers=1)
circ, status = htlg.tailor_logical_gate(stab_code_ldpc, connectivity_ldpc,logical_gate_ldpc, num_cz_layers=1, gurobi={"TimeLimit": 7200})

# Create a string buffer
buffer = io.StringIO()
sys.stdout = buffer  # Redirect stdout to buffer


# print the results

print(f'Status: "{status}"\n')
print(f'Circuit:\n{circ}')

# Restore stdout
sys.stdout = sys.__stdout__

# Write captured output to file
with open("circuit_H_1.txt", "w") as file:
    file.write(buffer.getvalue())