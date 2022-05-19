from permutations import PermutationGroup
import math


class Vector:
    '''Representation of a vector'''

    def __init__(self, values: list) -> None:
        if not type(values) == list:
            raise Exception('Not a list')
        check_number = lambda x: type(x) == int or type(x) == float
        valid = [check_number(x) for x in values]
        if not all(valid):
            raise Exception('Must be float or integer values only')
        self.values = values
    
    def __str__(self) -> str:
        '''Return string representation of vector'''
        return str(tuple(self.values))

    def __len__(self) -> int:
        '''Return dimension of vector'''
        return len(self.values)
    
    def __add__(self, y) -> 'Vector':
        '''Add two vectors'''
        sum_vector = []
        for a, b in zip(self.values, y.values):
            sum_vector.append(a+b)
        return Vector(sum_vector)
    
    def __sub__(self, y) -> 'Vector':
        '''Subtract one vector from another'''
        sub_vector = []
        for a, b in zip(self.values, y.values):
            sub_vector.append(a-b)
        return Vector(sub_vector)

    def __mul__(self, y) -> 'Vector':
        '''Multiply vector by a scalar'''
        # If y is a scalar
        if type(y) == int or type(y) == float:
            mul_vec = [value * y for value in self.values]
            return Vector(mul_vec)
        # If y is a vector
        if type(y) == Vector:
            if len(self) != len(y):
                raise Exception('Vectors not of equal length')
            mul_vec = [x_val * y_val for x_val, y_val in zip(self.values, y.values)]
            return sum(mul_vec)
        else:
            raise Exception('Scalar or vector types only')
    
    def __abs__(self) -> float:
        '''Return norm of vector'''
        square_sum = sum([value**2 for value in self.values])
        norm = math.sqrt(square_sum)
        return norm
    
    def __rmul__(self, y: 'Vector') -> 'Vector':
        '''Multiply scalar by vector'''
        return self.__mul__(y)
    
    def __matmul__(self, y: 'Vector') -> 'Vector':
        '''Return cross product of two vectors'''
        if type(y) != Vector:
            raise Exception('Not a vector')
        if len(self) != len(y):
            raise Expection('Unequal vector length')
        # Check that vectors have 3 dimensions since
        # the cross product only exists in 3 or
        # 7 Euclidean dimensions
        if len(self) != 3:
            raise Exception('Not a 3-dimensional vector')
        # Create matrix to get cross product
        unit_row = [1] * len(self)
        matrix = [unit_row, self.values, y.values]
        matrix = Matrix(matrix)
        # Compute cross product by calculating
        # determinant of ij-minors
        cross_product = []
        for i in range(len(self)):
            sign = (-1) ** i
            minor = matrix.minor(matrix.values, 0, i)
            cross_product.append(sign * minor.det())
        return Vector(cross_product)


class VectorList:

    def __init__(self, vector_list: list):
        # Need to validate
        self.vector_list = [Vector(v) for v in vector_list]
    
    def orthogonalise(self):
        '''Use Gram-Schmidt orthogonalisation
        to derive an orthonormal list.It is assumed
        that VectorList is independent and all vectors
        are of equal length
        '''
        w_list = []
        # Calulate first value of w
        w = self.vector_list[0]
        w_list.append(w)
        for i in range(1, len(self.vector_list)):
            # Create 0 vector 
            zero_vector = [0] * len(self.vector_list[0])
            w_sum = Vector(zero_vector)
            # Loop of j
            for j in range(0, i):
                # Compute dot product
                product = w_list[j] * self.vector_list[i]
                # Divide by square of norm
                product *= 1/(abs(w_list[j])**2)
                # Multiply by w
                product *= w_list[j]
                # Update w_sum
                w_sum += product
            # Compute w
            w = self.vector_list[i] - w_sum
            w_list.append(w)
        # Compute orthonormal list
        orthonormal_list = [u * (1/abs(u)) for u in w_list]
        return orthonormal_list


class Matrix:

    def __init__(self, values: list, ndigits=0) -> None:
        if type(values) != list:
            raise Exception('Not a list')
        # Check that all rows are lists
        check_rows = lambda x: type(x) == list
        valid = [check_rows(x) for x in values]
        if not all(valid):
            raise Exception('All rows must be lists')
        # Check that all rows are of equal length
        row_length = len(values[0])
        for row in values:
            if len(row) != row_length:
                raise Exception(
                    'All rows must be of equal length'
                )
        self.values = values
        self.ndigits = ndigits
    
    def __str__(self) -> str:
        '''Create string representation of
        matrix by organising into cols and rows
        '''
        max_rows = max([max(x) for x in self.values])
        adjustment = len(str(max_rows)) + 2
        string_rows = ''
        self = round(self, ndigits=self.ndigits)
        for row in self.values:
            string_row = ''
            for i in row:
                string_row += str(i).ljust(adjustment)
            string_rows += '\n' + string_row
        return string_rows
    
    def __len__(self) -> int:
        '''Return number of columns in matrix'''
        return len(self.values)
    
    def __mul__(self, y) -> 'Matrix':
        '''Multiply matrix with scalar'''
        # Check that y is a valid scalar
        if type(y) != int and type(y) != float:
            raise Exception('Not a scalar')
        product = []
        # Multiply each value by the scalar
        for i in range(self.col_num):
            row = []
            for j in range(self.row_num):
                row.append(self.row(i)[j] * y)
            product.append(row)
        return Matrix(product)
    
    def __rmul__(self, y) -> 'Matrix':
        '''Multiply scalar with matrix'''
        return self.__mul__(y)
    
    def __matmul__(self, y) -> 'Matrix':
        '''Multiply two matrices'''
        if self.row_num != y.col_num:
            raise Exception(
                'Matrices not in form A=nxm, B=mxp'
            )
        product_matrix = []
        for i in range(self.col_num):
            row = []
            for j in range(y.row_num):
                value_sum = 0
                for k in range(self.row_num):
                    product = self.row(i)[k] * y.row(k)[j]
                    value_sum += product
                row.append(value_sum)
            product_matrix.append(row)
        return Matrix(product_matrix)
    
    def __add__(self, y) -> 'Matrix':
        '''Add two matrices'''
        if type(y) != Matrix:
            raise Expection('Not of type Matrix')
        if len(self) != len(y):
            raise Expection('Not of equal length')
        valid = True
        for x_row, y_row in zip(self.values, y.values):
            if len(x_row) != len(y_row):
                valid = False
        if not valid:
            raise Expection('Not of equal dimensions')
        add_mul = []
        for x_row, y_row in zip(self.values, y.values):
            add_row = []
            for x_val, y_val in zip(x_row, y_row):
                add_row.append(x_val + y_val)
            add_mul.append(add_row)
        return Matrix(add_mul)
    
    def __round__(self, ndigits=0) -> 'Matrix':
        matrix = []
        for i in range(self.row_num):
            row = []
            for j in range(self.col_num):
                value = round(self.row(i)[j], ndigits)
                row.append(value)
            matrix.append(row)
        return Matrix(matrix)
    
    def index(self, i, j):
        return self.row(i)[j]
    
    def copy(self) -> 'Matrix':
        matrix = []
        for row in self.values:
            matrix.append(row)
        return Matrix(matrix, ndigits=self.ndigits)

    
    def dimension(self) -> tuple:
        '''Return number of columns and rows'''
        col_number = len(self.values)
        row_number = len(self.values[0])
        return col_number, row_number
    
    @property
    def row_num(self) -> int:
        '''Return number of rows'''
        return len(self.values[0])
    
    @property
    def col_num(self) -> int:
        '''Return number of columns'''
        return len(self.values)
    
    @property
    def is_square(self) -> bool:
        '''Determine whether matrix is square'''
        # col, row = self.dimension()
        if self.col_num == self.row_num:
            return True
        return False
    
    def column(self, j) -> list:
        '''Return column of matrix'''
        col = []
        for i in range(len(self)):
            value = self.values[i][j]
            col.append(value)
        return col

    def row(self, i) -> list:
        '''Return row of matrix'''
        return self.values[i]
    
    def replace_column(self, col_num: int, col_values: list) -> 'Matrix':
        '''Replace column of matrix with another'''
        matrix = []
        for i in range(self.row_num):
            new_row = self.row(i)
            new_row[col_num] = col_values[i]
            matrix.append(new_row)
        return Matrix(matrix)
    
    def replace_row(self, row_num: int, row_values: list) -> 'Matrix':
        '''Replace row of matrix with another'''
        matrix = []
        for i in range(self.col_num):
            if i == row_num:
                matrix.append(row_values)
            else:
                matrix.append(self.row(i))
        return Matrix(matrix)
    
    def det(self) -> float:
        '''Use sum formula to compute the determinant
        of a square matrix
        '''
        # Check that matrix is square
        if not self.is_square:
            raise Exception('Not a square matrix')
        # Get list of permutation objects of list
        # of integers 0,....,len(self)
        permutation_arg = list(range(len(self)))
        symmetric_group = PermutationGroup(permutation_arg)
        det = 0
        # Use sign formula to compute determinant
        for perm in list(symmetric_group.group()):
            sign = perm.sign()
            for count, elem in enumerate(perm.l):
                sign *= self.values[elem][count]
            det += sign
        return det
    
    def adjugate(self) -> 'Matrix':
        '''Compute adjugate of matrix'''
        adj = []
        for i in range(self.row_num):
            row = []
            for j in range(self.col_num):
                minor = self.minor(self.values, j, i)
                value = ((-1)**(i+j)) * minor.det()
                row.append(value)
            adj.append(row)
        return Matrix(adj)
    
    def transpose(self) -> 'Matrix':
        '''Compute transpose of matrix'''
        transpose = []
        for i in range(self.col_num):
            transpose.append(self.column(i))
        return Matrix(transpose)
    
    @property
    def is_invertible(self) -> bool:
        '''Determine whether a matrix
        is invertible
        '''
        # Must be a square matrix
        if not self.is_square:
            return False
        # Must have non-zero determinant
        if self.det() == 0:
            return False
        return True
    
    def is_upper_triangular(self, ndigits=0) -> bool:
        '''Determine whether matrix is upper
        triangular
        '''
        upper_triangular = True
        for i in range(self.row_num):
            for j in range(i):
                print(self.row(i)[j])
                if round(self.row(i)[j], ndigits=ndigits) != 0:
                    upper_triangular = False
        return upper_triangular
    
    def inverse(self) -> 'Matrix':
        '''Compute the inverse of the matrix'''
        if not self.is_invertible:
            raise Exception('Not invertible')
        det = 1/self.det()
        adj = self.adjugate()
        inv = det * adj
        return inv
    
    def identity(self) -> 'Matrix':
        '''Return identity matrix'''
        if not self.is_square:
            raise Exception('Not square')
        matrix = []
        for i in range(self.row_num):
            row = [0] * self.row_num
            row[i] = 1
            matrix.append(row)
        return Matrix(matrix)
    
    def diagonal_values(self) -> list:
        '''Return diagonal values'''
        values = []
        for i in range(self.row_num):
            for j in range(self.col_num):
                if i == j:
                    values.append(self.row(i)[j])
        return values
    
    @staticmethod
    def minor(mat, i: int, j: int) -> 'Matrix':
        '''Return ij-minor of matrix'''
        if type(i) != int or type(j) != int:
            raise Exception('Not integers')
        m = []
        for count_row, row in enumerate(mat):
            if count_row != i:
                m_row = []
                for count_val, value in enumerate(row):
                    if count_val != j:
                        m_row.append(value)
                m.append(m_row)
        return Matrix(m)
    
    def qr_decomposition(self) -> tuple:
        '''Return Q and R matrix using QR-decomposition
        where A=QR, Q is orthogonal and R is upper triangular
        '''
        # Get list of column vectors
        column_vectors = [self.column(i) for i in range(self.col_num)]
        column_vectors = VectorList(column_vectors)
        # Get orthonormal vectors
        orth_vectors = column_vectors.orthogonalise()
        # Compute Q matrix
        q = []
        for i in range(self.row_num):
            row = [vector.values[i] for vector in orth_vectors]
            q.append(row)
        q = Matrix(q)
        # R = Q^T * A
        r = q.transpose()@self
        return q, r
    
    def eigenvalues(self, iterations=100) -> list:
        '''Compute eigenvalues of matrix using 
        QR algorithm
        '''
        matrix = self.copy()
        # Get qr values and set matrix to r x q
        for i in range(iterations):
            q, r = matrix.qr_decomposition()
            matrix = r@q
        # Round values
        matrix = round(matrix, ndigits=self.ndigits)
        # Eigenvalues are diagonal entries of matrix
        return matrix.diagonal_values()
