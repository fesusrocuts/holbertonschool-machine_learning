#!/usr/bin/env python3
"""module add_matrices2D
file 5-across_the_planes.py
"""


def add_matrices2D(arr1, arr2):
    """
    fn add_matrices2D that sum each key-value with other matrix
    Args:
        arr1 (list of integers): The first parameter.
        arr2 (list of integers): The second parameter.
    Returns:
        arr3: The return new matrix for success, None otherwise.
    """
    try:
        arr3 = []
        if len(arr1) != len(arr2) or type(arr1) != type(arr2):
            raise Exception("Matrix are not have the same size")
        for i in range(len(arr1)):
            if len(arr1[i]) != len(arr2[i]):
                raise Exception("Matrix are not have the same size")
            if isinstance(arr1[i][0], (int, float)) is False:
                raise Exception("arr1 are not have type allow")
            if isinstance(arr1[i][1], (int, float)) is False:
                raise Exception("arr1 are not have type allow")
            if isinstance(arr2[i][0], (int, float)) is False:
                raise Exception("arr2 are not have type allow")
            if isinstance(arr2[i][1], (int, float)) is False:
                raise Exception("arr2 are not have type allow")
            arr3.append([arr1[i][0] + arr2[i][0],
                        arr1[i][1] + arr2[i][1]])
        return arr3
    except Exception as e:
        return None
