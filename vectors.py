from matrix import Matrix
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
            # Create 0 vector of length 
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
