class Vector:
    def __init__(self, elements, orientation='column', shape=None):
        self.elements = elements

        if shape is None and orientation == 'column':
            self.shape = (len(elements), 1)
        elif shape is None and orientation == 'row':
            self.shape = (1, len(elements))
        else:
            self.shape = shape

        self.orientation = orientation

    def dot(self, other):

        def verify_shape(a, b):
            if a.shape[1] != b.shape[0]:
                raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
            
        if isinstance(other, Vector):
            verify_shape(self, other)
            return sum(a * b for a, b in zip(self.elements, other.elements))
        
        elif isinstance(other, Matrix):
            verify_shape(self, other)

            if self.orientation == 'row':
                result = [sum(a * b for a, b in zip(self.elements, col)) for col in zip(*other.elements)]
                return Vector(result, 'row')
            
            elif self.orientation == 'column':
                result = [sum(a * b for a, b in zip(row, self.elements)) for row in other.elements]
                return Vector(result, 'column')
                  
    def transpose(self):
        if self.orientation == 'column':
            return Vector(self.elements, 'row', (1, len(self.elements)))
        elif self.orientation == 'row':
            return Vector(self.elements, 'column', (len(self.elements), 1))
    
    def concat(self, other):
        return Vector(self.elements + other.elements)
    
    def select_columns(self, indices):
            if isinstance(indices, int):
                return Vector([self.elements[indices]])
            elif isinstance(indices, list):
                return Vector([self.elements[i] for i in indices])
            return Vector([self.elements[i] for i in indices])

    def __str__(self):
        if self.orientation == 'column':
            return '\n'.join(map(str, self.elements))
        elif self.orientation == 'row':
            return '\t'.join(map(str, self.elements))
        
    def __add__(self, other):
        if isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            return Vector([a + b for a, b in zip(self.elements, other.elements)])
        else:
            raise TypeError("Unsupported type for addition")
        
    def __sub__(self, other):
        if isinstance(other, Vector):
            if self.shape != other.shape:
                raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
            return Vector([a - b for a, b in zip(self.elements, other.elements)])
        else:
            raise TypeError("Unsupported type for subtraction")
        
    def __len__(self):
        return len(self.elements)
    
    def __iter__(self):
        return iter(self.elements)
    
    def __round__(self, n=0):
        rounded_elements = []
        for element in self.elements:
            rounded_element = round(element, n)

            if isinstance(rounded_element, float) and n == 0 and rounded_element.is_integer():
                rounded_element = int(rounded_element)
            rounded_elements.append(rounded_element)

        return Vector(rounded_elements)
    
    def __neg__(self):
        return Vector([-element for element in self.elements])


class Matrix:
    def __init__(self, elements):
        self.elements = elements
        self.shape = (len(elements), len(elements[0]))

    def transpose(self):
        return Matrix([list(row) for row in zip(*self.elements)])

    def dot(self, other):

        def verify_shape(a, b):
            if a.shape[1] != b.shape[0]:
                raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
            
        if isinstance(other, Matrix):
            verify_shape(self, other)

            # Perform matrix multiplication
            result = [[sum(a * b for a, b in zip(row, col)) for col in zip(*other.elements)] for row in self.elements]
            return Matrix(result)

        elif isinstance(other, Vector):
            verify_shape(self, other)

            if other.orientation == 'column':
                result = [sum(a * b for a, b in zip(row, other.elements)) for row in self.elements]
                return Vector(result, 'column')
            else:
                transposed = self.transpose()
                result = [sum(a * b for a, b in zip(row, other.elements)) for row in transposed.elements]
                return Vector(result, 'row')

        else:
            raise TypeError("Unsupported type for dot product")

    def inv(self):
        n = len(self.elements)
        if any(len(row) != n for row in self.elements):
            raise ValueError("Only square matrices can be inverted.")

        A = [row[:] for row in self.elements]
        identity = [[float(i == j) for i in range(n)] for j in range(n)]

        for i in range(n):
            pivot = A[i][i]
            if pivot == 0:
                for j in range(i+1, n):
                    if A[j][i] != 0:
                        A[i], A[j] = A[j], A[i]
                        identity[i], identity[j] = identity[j], identity[i]
                        pivot = A[i][i]
                        break
                else:
                    raise ValueError("Matrix is singular and cannot be inverted.")

            for j in range(i, n):
                A[i][j] /= pivot
            for j in range(n):
                identity[i][j] /= pivot

            for k in range(n):
                if k != i:
                    factor = A[k][i]
                    for j in range(i, n):
                        A[k][j] -= factor * A[i][j]
                    for j in range(n):
                        identity[k][j] -= factor * identity[i][j]

        return Matrix(identity)
    
    def concat_right(self, other):
        # Check dims
        if self.shape[0] != other.shape[0]:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        
        if isinstance(other, Vector):
            new_elements = [self_row + [other_row] for self_row, other_row in zip(self.elements, other.elements)]
            return Matrix(new_elements)
        
        elif isinstance(other, Matrix):
            new_elements = [self_row + other_row for self_row, other_row in zip(self.elements, other.elements)]
            return Matrix(new_elements)
    
    def select_columns(self, indices):

        if len(indices) == 1:
            return Vector([row[indices[0]] for row in self.elements], 'column')

        new_elements = [[row[i] for i in indices] for row in self.elements]

        return Matrix(new_elements)
    
    def drop_columns(self, indices):
        new_elements = [[value for idx, value in enumerate(row) if idx not in indices] for row in self.elements]

        if not new_elements or not new_elements[0]:
            raise ValueError("Resulting matrix is empty after dropping columns.")

        return Matrix(new_elements)
    
    def to_vector(self):
        if self.shape[0] == 1:
            return Vector(self.elements[0], 'row')
        elif self.shape[1] == 1:
            return Vector([row[0] for row in self.elements], 'column')

    def __str__(self):
        return '\n'.join(['\t'.join(map(str, row)) for row in self.elements])


def identity_matrix(n):
    identity_elements = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    return Matrix(identity_elements)

def arange(start, stop, step=1):
    elements = []
    while start < stop:
        elements.append(start)
        start += step
    return Vector(elements)

class Simplex:
    def __init__(self, n, m, C, A, b) -> None:
        self.n = n # Number of constraints
        self.m = m # Number of variables
        self.C = Vector(C)
        self.A = Matrix(A)
        self.b = Vector(b)
        
    
    def test(self):
        # print all the variables
        print(f'n: {self.n}')
        print(f'm: {self.m}')

        print('*' * 50)

        print(f'A:\n{self.A}')
        print(f'A shape: {self.A.shape}')
        print(f'A transpose:\n{self.A.transpose()}')
        print(f'A transpose shape: {self.A.transpose().shape}')

        print('*' * 50)

        print(f'b:\n{self.b}')
        print(f'b shape: {self.b.shape}')
        print(f'b transpose:\n{self.b.transpose()}')
        print(f'b transpose shape: {self.b.transpose().shape}')

        print('*' * 50)

        print(f'C:\n{self.C}')
        print(f'C shape: {self.C.shape}')
        print(f'C transpose:\n{self.C.transpose()}')
        print(f'C transpose shape: {self.C.transpose().shape}')

        print('*' * 50)
        
        print(f'Dot product between A.T and b:\n{self.A.transpose().dot(self.b)}')
        print(f'Shape of it:\n{self.A.transpose().dot(self.b).shape}')

        print(f'A.T:\n{self.A.transpose()}')
        print(f'A.T shape:\n{self.A.transpose().shape}')

        print(f'b:\n{self.b}')
        print(f'b shape:\n{self.b.shape}')

        print('*' * 50)

        aux = Vector([0,1,2,3])
        print(f'aux:\n{aux}')
        print(f'selected 0 and 1 from aux:\n{aux.select_columns([0, 1])}')

        print('*' * 50)

        print('test matrix multiplication')

        aux = Matrix([[1,2,3], [4,5,6]])
        print(f'aux:\n{aux}')
        aux2 = Matrix([[1,2], [3,4], [5,6]])
        print(f'aux2:\n{aux2}')
        print(f'aux * aux2:\n{aux.dot(aux2)}')
        
        print('*' * 50)

        print('test vector multiplication')

        aux = Vector([1,2,3])
        print(f'aux:\n{aux}')
        aux2 = Vector([1,2,3], 'row')
        print(f'aux2:\n{aux2}')
        print(f'aux * aux2:\n{aux.dot(aux2)}')

        print('*' * 50)

        print('test vector and matrix multiplication')

        aux = Vector([1,2,3], 'row')
        print(f'aux:\n{aux}')
        aux2 = Matrix([[1,2,3], [4,5,6], [7,8,9]])
        print(f'aux2:\n{aux2}')

        print(f'aux * aux2:\n{aux.dot(aux2)}')

        print('¨' * 20)

        print(f'aux2 * aux.T:\n{aux2.dot(aux.transpose())}')


    def add_slack_vars(self):
        I = identity_matrix(self.n)
        A = self.A.concat_right(I)
        C = self.C.concat(Vector([0] * self.n))
        
        return A, C


    def remove_negative_b(self, A, b):
        alter = []
        for i in range(A.shape[0]):
            if b.elements[i] < 0:
                alter.append(i)
                A.elements[i] = [-x for x in A.elements[i]]
                b.elements[i] = -b.elements[i]
        return A, b, alter

    
    def solve(self, max_iterations=100, verbose=False):
        self.basis_size = self.n

        self.last_Xb = None
        self.last_basis = None
        self.last_B = None
        self.unlimited_column = None

        self.two_phase = None

        original_A = self.A
        original_b = self.b
        original_C = self.C

        # Put in std form

        # Add slack variables
        self.A, self.C = self.add_slack_vars()

        # Remove negative b
        self.A, self.b, rows_altered = self.remove_negative_b(self.A, self.b)

        # Change max to min to make simplex
        self.C = Vector([-x for x in self.C.elements])
        
        if len(rows_altered) > 0:
            self.two_phase = True
            response = self.two_phases_simplex(max_iterations=max_iterations, verbose=verbose)
        else:
            self.two_phase = False
            response = self.simplex(max_iterations=max_iterations, verbose=verbose)
            self.last_Xb = response[1]
            self.last_basis = response[2]
            self.last_B = response[3]
        
        if response[0] == 'optimal':
            print('otima')
            print(round(response[1], 7))
            print(*round(response[2], 7), sep=' ')
            print(*round(response[3], 7), sep=' ')
            
        elif response[0] == 'unbounded':
            print('ilimitada')

            # print(self.unlimited_column)
            # print(self.last_B)
            # print(self.last_Xb)
            # print('basis\n', self.last_basis)

            certificate1 = Vector([-1] * (self.m + self.n))

            for idx, base in enumerate(self.last_basis):
                certificate1.elements[base] = self.last_Xb.elements[idx]

            certificate1_real = []
            for i in range(m):
                if certificate1.elements[i] == -1:
                    certificate1.elements[i] = 0
                certificate1_real.append(certificate1.elements[i])

            certificate1_real = Vector(certificate1_real)

            # caso em que não há iterações
            if certificate1_real.elements == [0] * (self.m):
                certificate1 = Vector([-1] * (self.m + self.n))

                certificate1.elements[self.unlimited_column] = 0
                for idx, base in enumerate(self.last_basis):
                    certificate1.elements[base] = self.last_Xb.elements[idx]

                certificate1_real = []
                for i in certificate1:
                    if i >= 0:
                        certificate1_real.append(i)

                certificate1_real = Vector(certificate1_real)


            certificate2 = Vector([0] * (self.m + self.n))


            print(*round(certificate1_real, 7), sep=' ')
                    
        
        elif response[0] == 'infeasible':
            print('inviavel')
            print(*round(-response[3], 7), sep=' ')

        else:
            print('erro')
    

    def two_phases_simplex(self, max_iterations=100, verbose=False) -> tuple[str, Vector, Vector]:
    
        # Add n auxiliar variables
        I = identity_matrix(self.n)
        self.A = self.A.concat_right(I)
        
        real_C = self.C
        
        self.C = Vector([0] * (self.m + 2*self.n))
        
        for i in range(self.n):
            self.C.elements[-i-1] = 1
        
        # solve auxiliar problem
        simplex = self.simplex(max_iterations=max_iterations, verbose=verbose)

        if verbose:
            print(simplex)
            print('END AUXILIAR')
        
        if simplex[0] == 'optimal':
            self.last_Xb = simplex[4]
            self.last_basis = simplex[5]
            self.last_B = simplex[6]

        if simplex[0] == 'unbounded' or (simplex[0] == 'optimal' and simplex[1] != 0):
            return 'infeasible', simplex[1], simplex[2], simplex[3]

        # remove auxiliar variables
        num_auxiliar = self.n

        for _ in range(num_auxiliar):
            self.C.elements.pop()
            self.A = self.A.drop_columns([self.A.shape[1] - 1])
            
        self.C = real_C
        
        # solve original problem
        simplex = self.simplex(max_iterations=max_iterations, verbose=verbose)
        
        return simplex
        

    def simplex(self, max_iterations=100, verbose=False) -> tuple[str, Vector, Vector]:
        
        if verbose:
            print('\nSTART SIMPLEX\n')
            
            print(f'\nC:\n{self.C.transpose()}')
            print(f'\nA:\n{self.A}')
            print(f'\nb:\n{self.b}')
        
        # Define firsts basis and non-basis
        basis = arange(self.A.shape[1] - self.n, self.A.shape[1])
        
        num_iterations = 0
        while num_iterations <= max_iterations:
            num_iterations += 1
        
            # Calculate B
            # B shape = n x n
            B = self.A.select_columns(basis)
            B_inv = B.inv()

            # Calculate X_b
            X_b = B_inv.dot(self.b)

            # Calculate C_b
            C_b = self.C.select_columns(basis)

            # Calculate P^t = C_b^T * B^-1
            P_t = C_b.transpose().dot(B_inv)
            

            # Calculate S_b = C_b^T * B^-1 * A - C^T
            S = self.C.transpose() - P_t.dot(self.A)
            S = round(S, 5)
            
            if verbose:
                print(f'\n\nIteration {num_iterations}')
                print(f'\nBasis:\n{basis}')
                print(f'\nB:\n{B}')
                print(f'\nB^-1:\n{B_inv}')
                print(f'\nX_b:\n{X_b}')
                print(f'\nC_b:\n{C_b}')
                print(f'\nP_t:\n{P_t}')
                print(f'\nA:\n{self.A}')
                print(f'\nC:\n{self.C.transpose()}')
                print(f'\nS:\n{S}\n')

            # Check if problem is optimal
            if all(s >= 0 for s in S.elements):
                X_real = Vector([0] * self.m)
                for b in basis:
                    if b < self.m:
                        X_real.elements[b] = X_b.elements[basis.elements.index(b)]
                return 'optimal', -C_b.transpose().dot(X_b), X_real.transpose(), -P_t, X_b, basis, B
            
            # Find k such that S[k] < 0

            k = 0
            for i in range(len(S.elements)):
                if S.elements[i] < 0:
                    k = i
                    break

            # k = S.elements.index(min(S.elements))

            # Calculate Y = B^-1 * A_k
            Y = B_inv.dot(self.A.select_columns([k]))
            
            if verbose:
                print(f'\nk: {k}')
                print(f'\nA_k:\n{self.A.select_columns([k])}')
                print(f'\nY:\n{Y}\n')

            # Check if problem is unbounded
            if all(y <= 0 for y in Y.elements):
                for i in range(len(Y.elements)):
                    if Y.elements[i] < 0:
                        self.unlimited_column = i
                        break
                return ('unbounded', X_b, basis, B)
        
            # Minimum ratio test:
            ratios = []
            for x_b, y in zip(X_b.elements, Y.elements):
                if y > 0:
                    ratios.append(x_b / y)
                else:
                    ratios.append(float('inf'))
            l = ratios.index(min(ratios))

            # Update basis and non-basis
            basis.elements[l] = k

        if verbose:
            print('\nEND SIMPLEX\n')


def read_input(nome_do_arquivo):
    with open(nome_do_arquivo, 'r') as arquivo:
        n, m = map(int, arquivo.readline().split())
        costs = list(map(int, arquivo.readline().split()))

        rest = []
        for _ in range(n):
            restricao = list(map(int, arquivo.readline().split()))
            rest.append(restricao)

    return n, m, costs, rest

def separate_matrix(A):
    A_trimmed = [row[:-1] for row in A]

    # Create b by extracting the last column from the original matrix
    b = [row[-1] for row in A]

    return A_trimmed, b

if __name__ == '__main__':
    import sys

    filename = sys.argv[1]

    n, m, c, rest = read_input(filename)
    A, b = separate_matrix(rest)

    simplex = Simplex(n, m, c, A, b)
    simplex.solve(verbose=False)