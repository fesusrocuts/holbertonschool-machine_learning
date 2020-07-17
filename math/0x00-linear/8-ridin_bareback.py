#!/usr/bin/env python3
def mat_mul(mat1, mat2):
    try:
        if type(arr1) is not type(arr2):
            return None

        new_list = []
        if type(arr1) is list:
            for i in range(len(arr1)):
                new_list.append(arr1[i] + arr2[i])
        return new_list
    except Exception as e:
        return None
