#!/usr/bin/env python3
""" code mat_mul"""


def mat_mul(mat1, mat2):
    """ fn mat_mul"""
    if len(mat1[0]) != len(mat2):
        return None
    result = [[0 for x in mat2[0]] for row in mat1]
    for a in range(len(mat1)):
        for b in range(len(mat2[0])):
            for c in range(len(mat2)):
                result[a][b] += mat1[a][c] * mat2[c][b]
    return result
