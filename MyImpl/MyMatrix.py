import taichi as ti
import taichi_glsl as ts
import numpy as np
import tina


@ti.func
def make_shape(m):
    product = 1
    for s in ti.smart(m.shape):
        product *= s
    shape = tina.V(m.shape[0], product // m.shape[0])
    return shape
    #return m.shape[0], ti.select(len(m.shape) == 1, 1, m.shape[1])
    
'''Start mul'''
@ti.func
def mvmul_func(res, a, b):
    a_shape = res.shape#make_shape(a)
    for i in ti.ndrange(a_shape[0]):
        result = 0.
        for k in range(a_shape[1]):
            if a[i, k] != 0. and b[k] != 0.:
                result += a[i, k] * b[k]
        if result != 0.:
            res[i] = result

@ti.kernel
def mvmul_kernel(res : ti.template(), a : ti.template(), b : ti.template()):
    mvmul_func(res, a, b)

@ti.func
def matmul_func(res, a, b):
    res_shape = res.shape#make_shape(res)
    a_shape = a.shape#make_shape(a)
    for i, j in ti.ndrange(a_shape[0], res_shape[1]):
        result = 0.
        for k in range(a_shape[1]):
            if a[i, k] != 0. and b[k, j] != 0.:
                result += a[i, k] * b[k, j]
        if result != 0.:
            res[i, j] = result

@ti.kernel
def matmul_kernel(res : ti.template(), a : ti.template(), b : ti.template()):
    matmul_func(res, a, b)
'''Start mul'''
        
'''Start mul batch'''
@ti.func
def mvmul_batch_func(res, a, b):
    a_shape = a.shape#make_shape(a)
    for i in ti.ndrange(a_shape[0]):
        result = ti.Matrix([[0. for _ in range(res.m)] for _ in range(res.n)])
        for k in range(a_shape[1]):
            result += a[i, k] @ b[k]
        if any(result != 0.):
            res[i] = result

@ti.kernel
def mvmul_batch_kernel(res : ti.template(), a : ti.template(), b : ti.template()):
    mvmul_batch_func(res, a, b)
    
@ti.func
def matmul_batch_func(res, a, b):
    res_shape = res.shape#make_shape(res)
    a_shape = a.shape#make_shape(a)
    for i, j in ti.ndrange(a_shape[0], res_shape[1]):
        result = ti.Matrix([[0. for _ in range(res.m)] for _ in range(res.n)])
        for k in range(a_shape[1]):
            result += a[i, k] @ b[k, j]
        if any(result != 0.):
            res[i, j] = result

@ti.kernel
def matmul_batch_kernel(res : ti.template(), a : ti.template(), b : ti.template()):
    matmul_batch_func(res, a, b)
'''End mul batch'''
    
'''Start kronos'''
@ti.func
def mvmul_kronos_func(res, a, b, kron):
    a_shape = make_shape(a)
    for i in ti.ndrange(a_shape[0]):
        result = 0.
        for k in range(a_shape[1]):
            result += a[i, k] * b[k]
        if result != 0.:
            res[i] = kron * result

@ti.kernel
def mvmul_kronos_kernel(res : ti.template(), a : ti.template(), b : ti.template(), kron : ti.template()):
    mvmul_kronos_func(res, a, b, kron)
    
@ti.func
def matmul_kronos_func(res, a, b, kron):
    res_shape = res.shape#make_shape(res)
    a_shape = a.shape#make_shape(a)
    for i, j in ti.ndrange(a_shape[0], res_shape[1]):
        result = 0.
        for k in range(a_shape[1]):
            result += a[i, k] * b[k, j]
        if result != 0.:
            res[i, j] = kron * result

@ti.kernel
def matmul_kronos_kernel(res : ti.template(), a : ti.template(), b : ti.template(), kron : ti.template()):
    matmul_kronos_func(res, a, b, kron)
'''End kronos'''

'''Start add'''
@ti.func
def vecadd_batch_func(res, a, b, factor):
    res_shape = res.shape#make_shape(res)
    for i in ti.ndrange(res_shape[0]):
        result = a[i] + b[i] * factor
        if any(result != 0.):
            res[i] = result

@ti.kernel
def vecadd_batch_kernel(res : ti.template(), a : ti.template(), b : ti.template(), factor : ti.f32):
    vecadd_batch_func(res, a, b, factor)

@ti.func
def matadd_batch_func(res, a, b, factor):
    res_shape = res.shape#make_shape(res)
    for i, j in ti.ndrange(res_shape[0], res_shape[1]):
        result = a[i, j] + b[i, j] * factor
        if any(result != 0.):
            res[i, j] = result

@ti.kernel
def matadd_batch_kernel(res : ti.template(), a : ti.template(), b : ti.template(), factor : ti.f32):
    matadd_batch_func(res, a, b, factor)
'''End add'''
    
'''Start dot'''
@ti.func
def vecdot_func(res, a, b):
    a_shape = a.shape#make_shape(a)
    for i in ti.ndrange(a_shape[0]):
        res[None] += a[i] * b[i]

@ti.kernel
def vecdot_kernel(res : ti.template(), a : ti.template(), b : ti.template()):
    vecdot_func(res, a, b)

@ti.func
def matdot_func(res, a, b):
    a_shape = a.shape#make_shape(a)
    for i, j in ti.ndrange(a_shape[0], a_shape[1]):
        res[None] += a[i, j] * b[i, j]

@ti.kernel
def matdot_kernel(res : ti.template(), a : ti.template(), b : ti.template()):
    matdot_func(res, a, b)
    
@ti.func
def vecdot_batch_func(res, a, b):
    a_shape = a.shape#make_shape(a)
    for i in ti.ndrange(a_shape[0]):
        res[None] += a[i].dot(b[i])

@ti.kernel
def vecdot_batch_kernel(res : ti.template(), a : ti.template(), b : ti.template()):
    vecdot_batch_func(res, a, b)
    
@ti.func
def matdot_batch_func(res, a, b):
    a_shape = make_shape(a)
    for i, j in ti.ndrange(a_shape[0], a_shape[1]):
        res[None] += (a[i, j] * b[i, j]).sum()

@ti.kernel
def matdot_batch_kernel(res : ti.template(), a : ti.template(), b : ti.template()):
    matdot_batch_func(res, a, b)
'''End dot'''

@ti.func
def get_mat_ele(A, i, j, di, dj):
    i0 = i // di
    j0 = j // dj
    i1 = i - i0 * di
    j1 = j - j0 * dj
    return A[i0, j0][i1, j1]

@ti.func
def set_mat_ele(A, ele, i, j, di, dj):
    i0 = i // di
    j0 = j // dj
    i1 = i - i0 * di
    j1 = j - j0 * dj
    A[i0, j0][i1, j1] = ele
