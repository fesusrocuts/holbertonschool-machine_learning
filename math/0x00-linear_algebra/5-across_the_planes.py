#!/usr/bin/env python3
"""module add_matrices2D
file 5-across_the_planes.py
"""


def add_matrices2D(mat1, mat2):
    """
    fn add_matrices2D that sum each key-value with other matrix
    Args:
        mat1 (list of integers): The first parameter.
        mat2 (list of integers): The second parameter.
    Returns:
        mat3: The return new matrix for success, None otherwise.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    mat3 = []
    for row1, row2 in zip(mat1, mat2):
        mat3.append([])
        for x, y in zip(row1, row2):
            mat3[-1].append(x + y)
    return mat3
