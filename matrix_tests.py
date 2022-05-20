import random
import numpy as np
import pytest
from matrix import Matrix


# Generate random matrix
def random_matrix(col_num, row_num, upper=10, lower=-10) -> list:
    '''Generate random matrix'''
    matrix = []
    for i in range(col_num):
        row = [random.randint(lower, upper) for j in range(row_num)]
        matrix.append(row)
    return matrix


def get_np_det(matrix, round_value=5):
    # Get determinant of np matrix
    np_matrix = np.array(matrix)
    np_det = np.linalg.det(np_matrix)
    np_det = round(np_det, round_value)
    return np_det


def get_test_det(matrix, round_value=5):
    # Get determinant of test matrix
    test_matrix = Matrix(matrix)
    test_det = test_matrix.det()
    # Round determinant
    test_det = round(test_det, round_value)
    return test_det


def get_np_inv(matrix, round_value=5):
    # Get inverse of np matrix
    np_matrix = np.array(matrix)
    np_inv = np.linalg.inv(np_matrix)
    return np_inv


def get_test_inv(matrix, round_value=5):
    # Get inverse of test matrix
    test_matrix = Matrix(matrix)
    test_inv = test_matrix.inverse()
    return np.array(test_inv.values)


# Test determinant of square matrices


def test_2x2_determinant():
    # Get test matrix
    matrix = random_matrix(2, 2)
    assert get_np_det(matrix) == get_test_det(matrix)


def test_3x3_determinant():
    # Get test matrix
    matrix = random_matrix(3, 3)
    assert get_np_det(matrix) == get_test_det(matrix)


def test_4x4_determinant():
    # Get test matrix
    matrix = random_matrix(4, 4)
    assert get_np_det(matrix) == get_test_det(matrix)


def test_5x5_determinant():
    # Get test matrix
    matrix = random_matrix(5, 5)
    assert get_np_det(matrix) == get_test_det(matrix)


# Test inverse of square matrices


def test_2x2_inverse():
    # Get test matrix
    matrix = random_matrix(2, 2)
    np.testing.assert_array_almost_equal(
        get_np_inv(matrix),
        get_test_inv(matrix)
    )


def test_3x3_inverse():
    # Get test matrix
    matrix = random_matrix(3, 3)
    np.testing.assert_array_almost_equal(
        get_np_inv(matrix),
        get_test_inv(matrix)
    )


def test_4x4_inverse():
    # Get test matrix
    matrix = random_matrix(4, 4)
    np.testing.assert_array_almost_equal(
        get_np_inv(matrix),
        get_test_inv(matrix)
    )


def test_5x5_inverse():
    # Get test matrix
    matrix = random_matrix(5, 5)
    np.testing.assert_array_almost_equal(
        get_np_inv(matrix),
        get_test_inv(matrix)
    )
