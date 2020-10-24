#!/usr/bin/env python3
"""
fn the inverse of a function
"""


def inverse(matrix):
    """
    fn the inverse of a function
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for m in matrix:
        if not isinstance(m, list):
            raise TypeError("matrix must be a list of lists")
        if len(m) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1/(matrix[0][0])]]

    iM = adjugate(matrix)
    determinantNum = determinant(matrix)
    if determinantNum == 0:
        return None
    for i in range(len(iM)):
        for j in range(len(iM)):
            iM[i][j] = iM[i][j] / (determinantNum)
    return iM


def adjugate(matrix):
    """
    fn adjugate of given matrix
    """
    c = cofactor(matrix)
    return transpose_matrix(c)


def transpose_matrix(matrix):
    """
    fn transpose given matrix
    """
    return [[row[m] for row in matrix] for m in range(len(matrix[0]))]


def cofactor(matrix):
    """
    fn cofactor of a matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for m in matrix:
        if not isinstance(m, list):
            raise TypeError("matrix must be a list of lists")
        if len(m) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    cM = []
    for row in range(len(matrix)):
        nM = []
        for col in range(len(matrix[row])):
            sub_minr = [[matrix[i][j] for j in range(len(matrix))
                        if (j != col and i != row)]
                        for i in range(len(matrix))]
            sub_minr = [i for i in sub_minr if len(i) == len(matrix) - 1]

            nM.append((-1)**(row + col) * determinant(sub_minr))

        cM.append(nM)
    return cM


def minor(matrix):
    """
    fn minor of a matrix
    """
    # check if is list and
    # check if is square matrix
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for m in matrix:
        if not isinstance(m, list):
            raise TypeError("matrix must be a list of lists")
        if len(m) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    mM = []
    for row in range(len(matrix)):
        nM = []
        for col in range(len(matrix[row])):
            sub_minr = [[matrix[i][j] for j in range(len(matrix))
                        if (j != col and i != row)]
                        for i in range(len(matrix))]
            sub_minr = [i for i in sub_minr if len(i) == len(matrix) - 1]

            nM.append(determinant(sub_minr))

        mM.append(nM)
    return mM


def determinant(matrix):
    """
    fn determinant
    """
    det_returns = 0

    # check if is list and
    # check if is square matrix
    if not isinstance(matrix, list) or len(matrix) < 1:
        raise TypeError("matrix must be a list of lists")
    if ((len(matrix) == 1 and isinstance(matrix[0], list))
            and len(matrix[0]) == 0):
        return 1
    for m in matrix:
        if not isinstance(m, list):
            raise TypeError("matrix must be a list of lists")
        if len(m) != len(matrix[0]) or len(m) != len(matrix):
            raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]

    # working with 2X2 submatrices, then end
    if len(matrix) == 2 and len(matrix[0]) == 2:
        matrix_2X2 = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        return matrix_2X2

    # define submatric for each column
    for fCol in list(range(len(matrix))):
        matCp = matrix_cp(matrix)
        # remove first row
        matCp = matCp[1:]
        mheight = len(matCp)

        # remaining submatrix
        for idx in range(mheight):
            matCp[idx] = matCp[idx][0:fCol] + matCp[idx][fCol+1:]

        # signs for
        # submatrix multiplier
        sign = (-1) ** (fCol % 2)
        # pass recursively
        subMatrix_Det = determinant(matCp)
        det_returns += sign * matrix[0][fCol] * subMatrix_Det
    return det_returns


def onFillMatrixWithZero(rows, cols):
    """
    matrix filled with zeros RowsxCols
    """
    matrixZ = []
    while len(matrixZ) < rows:
        matrixZ.append([])
        while len(matrixZ[-1]) < cols:
            matrixZ[-1].append(0.0)
    return matrixZ


def matrix_cp(matrixA):
    """
    copy matrix
    """
    # dimensions
    cols = len(matrixA[0])
    rows = len(matrixA)
    
    # fill
    matrixCp = onFillMatrixWithZero(rows, cols)

    # copy values
    for i in range(rows):
        for j in range(cols):
            matrixCp[i][j] = matrixA[i][j]
    return matrixCp
