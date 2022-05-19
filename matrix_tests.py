import random
import numpy as np
from matrix import Matrix


# Generate random matrix
def random_matrix(col_num, row_num, upper=10, lower=-10) -> list:
    '''Generate random matrix'''
    matrix = []
    for i in range(col_num):
        row = [random.randint(lower, upper) for j in range(row_num)]
        matrix.append(row)
    return matrix


def test_det_matrix(matrix, round_value=5):
    '''Test determinant of matrix against np'''
    # Get determinant of np matrix
    np_matrix = np.matrix(matrix)
    np_det = np.linalg.det(np_matrix)
    # Get determinant of test matrix
    test_matrix = Matrix(matrix)
    test_det = test_matrix.det()
    # Round both determinants
    np_det = round(np_det, round_value)
    test_det = round(test_det, round_value)
    # Assert
    assert(np_det == test_det)


def test_inv_matrix(matrix, round_value=5):
    '''Test inverse of matrix against np'''
    # Get inverse of np matrix
    np_matrix = np.matrix(matrix)
    np_inv = np.linalg.inv(np_matrix)
    # Get inverse of test matrix
    test_matrix = Matrix(matrix)
    test_inv = test_matrix.inverse()
    # Convert test_inv to np matrix
    test_inv = np.matrix(test_inv.values)
    # Assert
    np.testing.assert_array_almost_equal(
        test_inv, np_inv, decimal=round_value)


# Generate matrices
def test_matrices(function, count, col_num, row_num):
    for i in range(count):
        matrix = random_matrix(col_num, row_num)
        function(matrix)

func_list = [test_inv_matrix, test_det_matrix]

# Test functions of square matrices
repeats = 5
for func in func_list:
    for i in range(2, 6):
        test_matrices(func, 5, i, i)
