#!/usr/bin/env python3
"""module add_matrices2D
file 5-across_the_planes.py
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    fn that concatenates two matrices along a specific axis
    Args:
        mat1 (list of integers): The first parameter.
        mat2 (list of integers): The second parameter.
        axis (int): 0 add rows into list, otherwise add cols into each row
    Returns:
        mat3: The return new matrix for success, None otherwise.
    """
    try:
        if axis == 0:
            # similary to add more rows
            mat3 = []
            list(map(lambda x: mat3.append(x), mat1))
            list(map(lambda x: mat3.append(x), mat2))
        if axis == 1:
            # similary to add more cols into element of the mat1
            # print("axis 1 ...")
            mat3 = [[], []]
            for i in range(len(mat2)):
                # print("mat2[{}]:".format(i))
                for i2 in mat1[i]:
                    # print("mat1[{}][{}]:".format(i,i2))
                    mat3[i].append(i2)
                mat3[i].append(mat2[i][0])
        return mat3
    except Exception as e:
        print(e)
        return None
