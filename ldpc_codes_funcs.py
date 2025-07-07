import numpy as np
import itertools
import warnings
from tqdm import tqdm

import networkx as nx
import math

import re

import stim

# GF(2) matrix operations
def gf2_matmul(A, B):
    return (A.dot(B) % 2)

def gf2_matadd(A, B):
    return (A ^ B)

def gf2_matpow(A, exp):
    # fast exponentiation mod2
    result = np.eye(A.shape[0], dtype=int)
    base = A.copy()
    e = exp
    while e > 0:
        if e & 1:
            result = gf2_matmul(result, base)
        base = gf2_matmul(base, base)
        e >>= 1
    return result

# RREF for GF2
def gf2_rref(A):
    M = A.copy().astype(int)
    rows, cols = M.shape
    pivot_cols = []
    r = 0
    for c in range(cols):
        # find pivot
        pivot = None
        for i in range(r, rows):
            if M[i, c] == 1:
                pivot = i
                break
        if pivot is None:
            continue
        # swap
        M[[r, pivot]] = M[[pivot, r]]
        pivot_cols.append(c)
        # eliminate
        for i in range(rows):
            if i != r and M[i, c] == 1:
                M[i, :] ^= M[r, :]
        r += 1
        if r == rows:
            break
    return M, pivot_cols

# Nullspace basis for GF2
def gf2_nullspace(A):
    M_rref, pivots = gf2_rref(A)
    rows, cols = M_rref.shape
    free_vars = [c for c in range(cols) if c not in pivots]
    basis = []
    for free in free_vars:
        v = np.zeros(cols, dtype=int)
        v[free] = 1
        # back-substitute pivot rows
        for i, pc in enumerate(pivots):
            v[pc] = M_rref[i, free]
        basis.append(v)
    return np.array(basis)

# Rowspace basis for GF2
def gf2_rowspace(A):
    R, pivots = gf2_rref(A)
    return R[:len(pivots), :]

# Solve M x = b over GF2, return one solution or None
def gf2_solve(M, b):
    M = M.copy().astype(int)
    b = b.copy().astype(int)
    rows, cols = M.shape
    # augmented
    Aug = np.concatenate([M, b.reshape(-1, 1)], axis=1)
    R, pivots = gf2_rref(Aug)
    # check consistency
    for i in range(len(R)):
        if np.all(R[i, :cols] == 0) and R[i, cols] == 1:
            return None
    # back-substitute
    x = np.zeros(cols, dtype=int)
    for i in range(len(pivots)-1, -1, -1):
        pc = pivots[i]
        x[pc] = R[i, cols]
        # subtract contributions of free vars
        for c in range(pc+1, cols):
            if R[i, c] == 1:
                x[pc] ^= x[c]
    return x

# test if v is in the rowspace spanned by rows of M_rows
def in_rowspace(M_rows, v):   
    # solve M_rows^T * a = v
    M = M_rows.copy().T
    b = v.copy()
    # augmented RREF
    A = np.concatenate((M, b[:,None]), axis=1)
    R, piv = gf2_rref(A)
    cols = M.shape[1]
    # inconsistency?
    for row in R:
        if not row[:cols].any() and row[cols]:
            return False
    return True

class TB_LDPC_code:
    def __init__(self,l,m):
        self.l = l
        self.m = m
        self.n = 2*m*l
        self.S_l = self.shift_matrix(l)
        self.S_m = self.shift_matrix(m)
        self.x = np.kron(self.S_l, np.eye(m, dtype=int))
        self.y = np.kron(np.eye(l, dtype=int), self.S_m)
        self.z = np.kron(self.S_l, self.S_m)
        self.period_z = np.lcm(l, m)

    # Generate cyclic shift matrix of size n
    def shift_matrix(self,n):
        S = np.zeros((n, n), dtype=int)
        for i in range(n):
            S[i, (i + 1) % n] = 1
        return S

    # code below is for a random code:
    def pick_terms(self, weight):
        terms = set()
        while len(terms) < weight:
            var = np.random.choice(['x', 'y', 'z'])
            if var == 'x': exp = np.random.randint(1, self.l)//2
            elif var == 'y': exp = np.random.randint(1, self.m)
            else: exp = np.random.randint(1, self.period_z-1)
            term = (var, exp)
            terms.add(term)
        return list(terms)

    # build A and B
    def build_sum(self,terms):
        mats = []
        for var, exp in terms:
            if var == 'x': mats.append(gf2_matpow(self.x, exp))
            elif var == 'y': mats.append(gf2_matpow(self.y, exp))
            else: mats.append(gf2_matpow(self.z, exp))
        # sum in GF2
        M = np.zeros_like(mats[0])
        for mat in mats:
            M ^= mat
        return M

    # generate random A,B
    def generate_stabilizers(self, 
                             W_A, 
                             W_B, 
                             A_terms = None, 
                             B_terms = None):
        if A_terms == None:
            self.A_terms = self.pick_terms(W_A)
        else:
            self.A_terms = A_terms
        if B_terms == None:
            self.B_terms = self.pick_terms(W_B)
        else:
            self.B_terms = B_terms
        self.W_A = W_A
        self.W_B = W_B
        
        self.A = self.build_sum(self.A_terms)
        self.B = self.build_sum(self.B_terms)
        
        # Check that A and B commutes 
        assert np.all(gf2_matmul(self.A, self.B) == gf2_matmul(self.B, self.A)), "A and B must commute"
        
        self.H_X = np.concatenate((self.A, self.B), axis=1) % 2
        self.H_Z = np.concatenate((self.B.T, self.A.T), axis=1) % 2

        # dimension of intersection ker(A) \cap ker(B)
        stacked = np.vstack([self.A, self.B])
        dim_inter = gf2_nullspace(stacked).shape[0]
        self.k = 2 * dim_inter

        return self.H_X, self.H_Z

    def find_logical_Z(self) -> np.ndarray:
        self.generate_css_logical_ops_symplectic()
        return self.Z_ops

    def find_logical_X(self) -> np.ndarray:
        self.generate_css_logical_ops_symplectic()
        return self.X_ops

    def generate_css_logical_ops_symplectic(self):
        """
        Generate paired logical X and Z operators for a CSS code using symplectic Gram-Schmidt.
        """
        H_X = self.H_X
        H_Z = self.H_Z
        n = self.n
        k = self.k
    
        # Step 1: Construct stabilizer generators from rowspace
        rsX = gf2_rowspace(H_X)
        rsZ = gf2_rowspace(H_Z)
    
        stab_generators = []
        for row in rsX:
            stab_generators.append(np.concatenate([row, np.zeros(n, dtype=int)]))  # X | 0
        for row in rsZ:
            stab_generators.append(np.concatenate([np.zeros(n, dtype=int), row]))  # 0 | Z
        stab_generators = np.array(stab_generators)
    
        # Step 2: Construct symplectic orthogonality constraints
        def symplectic_constraints(S, n):
            constraints = []
            for s in S:
                a, b = s[:n], s[n:]
                row = np.concatenate([b, a])  # symplectic inner product: x·b + z·a
                constraints.append(row)
            return np.array(constraints) % 2
    
        constraints = symplectic_constraints(stab_generators, n)
        centralizer = gf2_nullspace(constraints)
    
        # Step 4: Remove stabilizer vectors from centralizer
        def quotient_space_basis(centralizer, stabilizers):
            basis = []
            current = stabilizers.copy()
        
            for vec in centralizer:
                test = np.vstack((current, vec.reshape(1, -1)))
                rank_before = len(gf2_rref(current)[1])
                rank_after = len(gf2_rref(test)[1])
                if rank_after > rank_before:
                    basis.append(vec)
                    current = np.vstack((current, vec.reshape(1, -1)))
                if len(basis) == 2 * self.k:
                    break
            return np.array(basis)
        
        logical_space = quotient_space_basis(centralizer, stab_generators)
        assert logical_space.shape[0] >= 2 * k, "Centralizer too small for symplectic pairing"
    
        # Step 5: Symplectic Gram-Schmidt
        def symplectic_inner(u, v):
            return (np.dot(u[:n], v[n:]) + np.dot(u[n:], v[:n])) % 2
    
        X_ops, Z_ops = [], []
        used = np.zeros(len(logical_space), dtype=bool)
    
        for _ in range(k):
            found = False
            for i in range(len(logical_space)):
                if used[i]:
                    continue
                for j in range(i + 1, len(logical_space)):
                    if used[j]:
                        continue
                    if symplectic_inner(logical_space[i], logical_space[j]) == 1:
                        xi = logical_space[i]
                        zi = logical_space[j]
    
                        # Orthogonalize remaining vectors
                        for l in range(len(logical_space)):
                            if l in (i, j) or used[l]:
                                continue
                            v = logical_space[l]
                            if symplectic_inner(xi, v):
                                logical_space[l] = (v + zi) % 2
                            if symplectic_inner(zi, v):
                                logical_space[l] = (v + xi) % 2
    
                        X_ops.append(xi[:n])
                        Z_ops.append(zi[n:])
                        used[i] = used[j] = True
                        found = True
                        break
                if found:
                    break
            if not found:
                raise RuntimeError("Failed to find enough symplectic pairs")
    
        self.X_ops = np.array(X_ops)
        self.Z_ops = np.array(Z_ops)
        return self.X_ops, self.Z_ops

    def distance_from_logical_X(self) -> int:
        """
        Given a basis X_ops of k logical-X generators (shape (k, n)),  
        compute the code distance d_X = min weight(v) over all nonzero
        v in span(X_ops).
    
        This is an O(2^k) loop instead of O(2^n).
        """
        try:
            self.X_ops
        except AttributeError:
            self.find_logical_X()
        k = len(self.X_ops)
            
        # convert each basis row to a Python int mask
        basis_masks = []
        for row in self.X_ops:
            m = 0
            for j, bit in enumerate(row):
                if bit:
                    m |= 1 << j
            basis_masks.append(m)
    
        d = self.n+1
        # enumerate all non-zero combinations
        for mask in range(1, 1 << k):
            v = 0
            # XOR together the chosen basis masks
            for i in range(k):
                if (mask >> i) & 1:
                    v ^= basis_masks[i]
            w = v.bit_count()
            if w < d:
                d = w
                if d == 1:
                    return 1
        self.d = d
        return self.d

    def distance_from_logical_Z(self) -> int:
        """
        Given a basis Z_ops of k logical-Z generators (shape (k, n)),  
        compute the code distance d_Z = min weight(v) over all nonzero
        v in span(Z_ops).
    
        This is an O(2^k) loop instead of O(2^n).
        """
        try:
            self.Z_ops
        except AttributeError:
            self.find_logical_Z()
        k = self.k
            
        # convert each basis row to a Python int mask
        basis_masks = []
        for row in self.Z_ops:
            m = 0
            for j, bit in enumerate(row):
                if bit:
                    m |= 1 << j
            basis_masks.append(m)
    
        d = self.n+1
        # enumerate all non-zero combinations
        for mask in range(1, 1 << k):
            v = 0
            # XOR together the chosen basis masks
            for i in range(k):
                if (mask >> i) & 1:
                    v ^= basis_masks[i]
            w = v.bit_count()
            if w < d:
                d = w
                if d == 1:
                    return 1
        self.d = d
        return self.d

    def distance_from_logicals(self):
        distance_X = self.distance_from_logical_X()
        distance_Z = self.distance_from_logical_Z()
        self.d = min(distance_X, distance_Z)
        return self.d

    def find_random_tb_code(
        self,
        W_A: int, 
        W_B: int,
        k_min: int = 2,
        d_min: int = 5,
        max_trials: int = 10000,
        seed: int = None
    ):
        """
        Search for a random TB code with parameters k >= k_min and d >= d_min.
    
        Args:
            k_min: minimum logical qubits (k) desired.
            d_min: minimum distance (d) desired.
            max_trials: maximum number of random draws before giving up.
            seed: random seed for reproducibility.
    
        This function builds random A,B sums in GF(2)[G3], forms stabilizer matrices H_X,H_Z,
        computes k via nullspace dimension of [A;B], then finds distance
        """
        if seed is not None:
            np.random.seed(seed)
    
        
        for trial in tqdm(range(max_trials), 'trails'):
            try:
                del self.H_Z
            except AttributeError:
                1
            try:
                del self.H_X
            except AttributeError:
                2
            try:
                del self.Z_ops
            except AttributeError:
                3
            
            self.generate_stabilizers(W_A, W_B)
            n = self.n
            k = self.k
            l = self.l
            m = self.m
            
            # quick geometry checks (optional user constraints)
            Bt = self.B.T
            if Bt[:n//l,n//l:n//(2*l)].sum()>0: continue
            if np.abs(np.diff(Bt,axis=1)).sum()>3*m*l: continue
            if np.abs(np.diff(self.B,axis=1)).sum()>3*m*l: continue
    
            if k != k_min:
                continue

            self.generate_css_logical_ops_symplectic()
    
            d = self.distance_from_logicals()
            if d != d_min:
                continue
                
            # success
            print(n,k,d)
            return self.A_terms, self.B_terms
    
        raise RuntimeError(f"No code found in {max_trials} trials with k>= {k_min}, d>= {d_min}")

    
    def build_stim_error_correction_circuit(self,
                                            p: float = 0.01,
                                            rounds: int = 1,
                                            mem_type: str = 'Z'
    ) -> stim.Circuit:
        """
        Build a Stim circuit for repeated CSS error-correction cycles using CZ gates.
    
        This implements `rounds` rounds of syndrome extraction on data and ancilla qubits.
    
        Qubit indexing:
          - Data: 0..n-1
          - Z ancillas: n..n+r_Z-1 (r_Z = H_Z.shape[0])
          - X ancillas: n+r_Z..n+r_Z+r_X-1 (r_X = H_X.shape[0])
    
        Args:
            mem_type: which basis to memory-stabilize ('Z' or 'X').
    
        Returns:
            A Stim Circuit with:
            - QUBIT_COORDS for layout.
            - Reset of all qubits.
            - Rounds of: ancilla prep; four CZ cycles with error channels; ancilla measurement.
            - Detection events after each round and final data measurement.
        """
        r_Z = self.H_Z.shape[0]
        r_X = self.H_X.shape[0]
        total_qubits = self.n + r_Z + r_X
        circuit = stim.Circuit()
    
        n = self.n
        l = self.l
        m = self.m
    
        try:
            G = self.G
            pos = self.pos
        except AttributeError:
            G, pos = generate_tanner_graph(self)
        
        try:
            gates = self.gates
        except AttributeError:
            gates = generate_cz_gates(self)
    
        try:
            Z_ops = self.Z_ops
        except AttributeError:
            Z_ops = self.find_logical_Z()
    
        try:
            X_ops = self.X_ops
        except AttributeError:
            X_ops = self.find_logical_X()
    
        #--- Qubit coordinates ---#
        for q in range(n):
            x, y = pos[('d', q)]
            circuit.append('QUBIT_COORDS', [q], [x, y])
        for i in range(r_Z):
            idx = n + i
            x, y = pos[('Z', i)]
            circuit.append('QUBIT_COORDS', [idx], [x, y])
        for j in range(r_X):
            idx = n + r_Z + j
            x, y = pos[('X', j)]
            circuit.append('QUBIT_COORDS', [idx], [x, y])
    
        #--- Initialization ---#
        # Reset all qubits into |0> (Z basis)
        for q in range(n):
            circuit.append('R', [q])  
        for q in range(n):
            circuit.append('X_ERROR', [q], p)
        for q in range(r_Z+r_X):
            circuit.append('R', [q+n])  
        for q in range(r_Z+r_X):
            circuit.append('X_ERROR', [q+n], p)
        circuit.append('TICK')
    
        # If storing in X basis, rotate data into X after reset
        if mem_type.upper() == 'X':
            data_qubits = list(range(n))
            circuit.append('H', data_qubits)
            circuit.append('TICK')
    
        #--- Syndrome extraction rounds ---#
        for r in range(rounds):
            # 1) Ancilla preparation: H on all ancillas
            # for i in range(r_Z):
            #     circuit.append('H', [n + i])
            # for i in range(r_Z):
            #     circuit.append('DEPOLARIZE1', [n + i],p)
            # circuit.append('TICK')
            for j in range(r_X):
                circuit.append('H', [n + r_Z + j])
            for j in range(r_X):
                circuit.append('DEPOLARIZE1', [n + r_Z + j],p)
            circuit.append('TICK')
    
            # 2) Four CZ cycles per round
            for cycle in gates:
                # (a) Pre-CZ: apply H to data for X-ancillas
                # data_to_h = []
                # for gate in cycle:
                #     _, d, a = gate.split('-')
                #     if a[0] == 'X':
                #         data_to_h.append(int(d[1:]))
                # if data_to_h:
                #     circuit.append('H', data_to_h)
                #     circuit.append('DEPOLARIZE1', data_to_h, p)
    
                # (b) CZ interactions
                cz_targets = []
                cx_targets = []
                for gate in cycle:
                    _, d, a = gate.split('-')
                    data_idx = int(d[1:])
                    if a[0] == 'Z':
                        anc_idx = n + int(a[1:])
                        cz_targets += [data_idx, anc_idx]
                    else:
                        anc_idx = n + r_Z + int(a[1:])
                        cx_targets += [anc_idx, data_idx]
                    
                circuit.append('CX', cz_targets)
                circuit.append('CX', cx_targets)
                circuit.append('DEPOLARIZE2', cz_targets, p)
                circuit.append('DEPOLARIZE2', cx_targets, p)
    
                # (c) Post-CZ: undo H on same data
                # if data_to_h:
                #     circuit.append('H', data_to_h)
                #     circuit.append('DEPOLARIZE1', data_to_h, p)
                circuit.append('TICK')
                
    
            # 3) Ancilla measurement:
            # for i in range(r_Z):
            #     circuit.append('H', [n + i])
            for j in range(r_X):
                circuit.append('H', [n + r_Z + j])
            for j in range(r_X):
                circuit.append('DEPOLARIZE1', [n + r_Z + j],p)
            circuit.append('TICK')
            for i in range(r_Z):
                circuit.append('X_ERROR', [n + i], p)  # bit-flip error before measure
                circuit.append('MR',      [n + i])    # pure Z-basis measure
                circuit.append('X_ERROR', [n + i], p)  # bit-flip error before measure
            for j in range(r_X):
                circuit.append('X_ERROR', [n + r_Z + j], p)
                circuit.append('MR',      [n + r_Z + j])  # X-basis measure
                circuit.append('X_ERROR', [n + r_Z + j], p)
            
            # 4) Detection events
            if r == 0:
                # single-shot detectors for all ancillas
                if mem_type=='Z':
                    i = 0
                    coords = list(pos[('Z', i)]) + [r]
                    for i in range(r_Z):
                        coords = list(pos[('Z', i)]) + [r]
                        circuit.append('DETECTOR', [f'rec[-{r_X + r_Z - i}]'], coords)
                elif mem_type=='X':
                    for j in range(r_X):
                        coords = list(pos[('X', j)]) + [r]
                        circuit.append('DETECTOR', [f'rec[-{r_X - j}]'], coords)
            else:
                # detectors compare outcomes between rounds
                for i in range(r_Z):
                    coords = list(pos[('Z', i)]) + [r]
                    circuit.append('DETECTOR', [f'rec[-{r_X + r_Z - i}]', f'rec[-{2*r_Z + 2*r_X - i}]'], coords)
                for j in range(r_X):
                    coords = list(pos[('X', j)]) + [r]
                    circuit.append('DETECTOR', [f'rec[-{r_X - j}]', f'rec[-{r_Z + 2*r_X - j}]'], coords)
            if r<rounds-1:
                circuit.append('TICK')
    
        #--- Final data measurement and observables ---# and observables ---#
    
        # Include logical observables
        if mem_type=='Z':
            # Measure data qubits
            for q in range(n):
                circuit.append('X_ERROR', [q],p)
            for q in range(n):
                circuit.append('M', [q])
        
            # Detect data–ancilla correlations for final checks
            for i, Z in enumerate(self.H_Z):
                qubits = list(np.nonzero(Z)[0])
                coords = list(pos[('Z', i)]) + [rounds]
                recs = [f'rec[-{n - q}]' for q in qubits] + [f'rec[-{r_X + r_Z + n - i}]']
                circuit.append('DETECTOR', recs, coords)
                
            # Include logical observables
            for idx, Z in enumerate(Z_ops):
                qs = list(np.nonzero(Z)[0])
                circuit.append('OBSERVABLE_INCLUDE', [f'rec[-{n - q}]' for q in qs], idx)
        elif mem_type=='X':
            # Measure data qubits
            # for q in range(n):
                # circuit.append('H', [q])
            for q in range(n):
                circuit.append('Z_ERROR', [q],p)
            for q in range(n):
                circuit.append('MX', [q])
        
            # Detect data–ancilla correlations for final checks
            for i, X in enumerate(self.H_X):
                qubits = list(np.nonzero(X)[0])
                coords = list(pos[('X', i)]) + [rounds]
                recs = [f'rec[-{n - q}]' for q in qubits] + [f'rec[-{r_X + n - i}]']
                circuit.append('DETECTOR', recs, coords)
                
            # Include logical observables
            for idx, X in enumerate(X_ops):
                qs = list(np.nonzero(X)[0])
                circuit.append('OBSERVABLE_INCLUDE', [f'rec[-{n - q}]' for q in qs], idx)
    
        return circuit


def generate_tanner_graph(code):
    H_X = code.H_X
    H_Z = code.H_Z 
    l = code.l
    m = code.m 
    W = code.W_A + code.W_B
    n = H_X.shape[1]
    assert H_Z.shape[1] == n, "H_X and H_Z must have same number of columns"
    G = nx.Graph()
    pos = {}
    used_faces = set()
    # Data qubit nodes in grid
    for q in range(n):
        if q<n//2:
            row = 2*l - 2*(q // (m))
            col = 2*(q % (m)) 
        else:
            row = 2*l- 1 - 2*((q-n//2) // (m))
            col = 2*((q-n//2) % (m)) +1 
        node = ('d', q)
        G.add_node(node, bipartite=0)
        pos[node] = (col, row)
    

    # Check nodes: assign to nearest free face center
    for kind, H in [('X', H_X), ('Z', H_Z)]:
        for i in range(H.shape[0]):
            cnode = (kind, i)
            G.add_node(cnode, bipartite=1)
            # find data qubits this check connects
            connected = [j for j in range(n) if H[i, j]]
            
            pos[cnode] = (0,0)
            if kind == 'Z':
                if (connected[0]+connected[1]) % m == m-1:
                    col, row = pos[('d', connected[1])]
                    pos[cnode] = (col+1,row)
                else:
                    nearest_data = connected[1]
                    col, row = pos[('d', nearest_data)]
                    pos[cnode] = (col-1,row)
            if kind == 'X':
                if (connected[W-2]+connected[W-1]-2*m) % m == m-1:
                    col, row = pos[('d', connected[W-2])]
                    pos[cnode] = (col-1,row)
                else:
                    nearest_data = connected[W-1]
                    col, row = pos[('d', nearest_data)]
                    pos[cnode] = (col-1,row)
            
            # connect edges
            for j in connected:
                G.add_edge(cnode, ('d', j))
    code.pos = pos
    code.G = G
    return G, pos

def generate_cz_gates(code, Z_order=[0,3,2,1], X_order=[0,3,2,1]):
    """
    Construct CZ-based measurement cycles for a CSS code from its parity-check matrices.

    The code performs 4 "cycles" of CZ interactions between data qubits and X- or Z-type ancillas:
    - On even cycles, it measures Z-checks first; on odd cycles, X-checks first.
    - It alternates scanning the first half and second half of qubits to avoid conflicts.
    - Each cycle selects at most one CZ gate per ancilla and per data qubit.

    Returns:
        List[List[str]]: A list of 4 sublists, one per cycle, each containing gate strings
                         of the form 'CZ-D{data_index}-Z{check_index}' or 'CZ-D{data_index}-X{check_index}'.
    """
    H_X = code.H_X
    H_Z = code.H_Z 
    l = code.l
    m = code.m 
    W = code.W_A + code.W_B
    assert W==4, "Only implemented for weight-4 codes" 
    n = H_X.shape[1]
    gates = []

    pattern = re.compile(r"^CZ-D(\d+)-[XZ](\d+)$")

    # Perform 4 rounds (cycles) of CZ interactions
    for cycle in range(4):
        cycle_gates = []
        # Track which data qubits have been used in this cycle
        used_data = set()

        # Z-check interactions
        for i, row in enumerate(H_Z):
            connected = [j for j in range(n) if H_Z[i,j]]
            
            # Pick data half depending on cycle parity
            if Z_order[cycle] == 0:
                if (connected[0]+connected[1]) % m == m-1:
                    q = connected[0]
                else:
                    q = connected[1]
                gate = f"CZ-D{q}-Z{i}"
                cycle_gates.append(gate)
            elif Z_order[cycle] == 1:
                if (connected[0]+connected[1]) % m == m-1:
                    q = connected[1]
                else:
                    q = connected[0]
                gate = f"CZ-D{q}-Z{i}"
                cycle_gates.append(gate)
            elif Z_order[cycle] == 2:
                q = connected[2] 
                gate = f"CZ-D{q}-Z{i}"
                if q in used_data:
                    q = connected[3] 
                    gate = f"CZ-D{q}-Z{i}"
                used_data.add(q)
                cycle_gates.append(gate)
            elif Z_order[cycle] == 3:
                q = connected[3] 
                gate = f"CZ-D{q}-Z{i}"
                if q in used_data:
                    q = connected[2] 
                    gate = f"CZ-D{q}-Z{i}"
                used_data.add(q)
                cycle_gates.append(gate)

        # X-check interactions
        for i, row in enumerate(H_X):
            connected = [j for j in range(n) if H_X[i,j]]
            
            # Pick data half depending on cycle parity
            if X_order[cycle] == 0:
                if (connected[2]+connected[3]-2*m) % m == m-1:
                    q = connected[2]
                else:
                    q = connected[3]
                gate = f"CZ-D{q}-X{i}"
                cycle_gates.append(gate)
            elif X_order[cycle] == 1:
                if (connected[2]+connected[3]-2*m) % m == m-1:
                    q = connected[3]
                else:
                    q = connected[2]
                gate = f"CZ-D{q}-X{i}"
                cycle_gates.append(gate)
            elif X_order[cycle] == 2:
                q = connected[0] 
                gate = f"CZ-D{q}-X{i}"
                if q in used_data:
                    q = connected[1] 
                    gate = f"CZ-D{q}-X{i}"
                used_data.add(q)
                cycle_gates.append(gate)
            elif X_order[cycle] == 3:
                q = connected[1] 
                gate = f"CZ-D{q}-X{i}"
                if q in used_data:
                    q = connected[0] 
                    gate = f"CZ-D{q}-X{i}"
                used_data.add(q)
                cycle_gates.append(gate)

            

        gates.append(cycle_gates)
    code.gates = gates
    return gates

def gate_coordinates(s,pos):
    parts = s.split('-')[1:]       # e.g. ['D0','Z0'] or ['D5','X1']
    qubits = []
    for p in parts:
        kind = p[0]
        idx  = int(p[1:])
        if kind == 'D':
            kind = 'd'
        qubits.append(np.array(pos[(kind, idx)]))
    return np.array(qubits)