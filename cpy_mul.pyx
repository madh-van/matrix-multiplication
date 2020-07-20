cpdef list cpy_mul(list X, list Y):
    cdef int x
    cdef int y
    cdef list X_row
    cdef tuple Y_col
    cdef int total
    cdef list result = []
    for X_row in X:
        result.append([])
        for Y_col in zip(*Y):
            total = 0
            for x, y in zip(X_row,Y_col):
                total += x * y
            result[-1].append(total)
    return result
