#!/usr/bin/env python3
"""module add_matrices2D
file 5-across_the_planes.py
"""


def cat_arrays(arr1, arr2):
    """
    fn cat_arrays append the values to other matrix
    Args:
        arr1 (list of integers): The first parameter.
        arr2 (list of integers): The second parameter.
    Returns:
        arr3: The return new matrix for success, None otherwise.
    """
    try:
        arr3 = []
        list(map(lambda x: arr3.append(x), arr1))
        list(map(lambda x: arr3.append(x), arr2))
        return arr3
    except Exception as e:
        return None
