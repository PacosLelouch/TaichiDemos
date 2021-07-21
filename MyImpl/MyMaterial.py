import taichi as ti
import numpy as np
import tina
import taichi_glsl as ts

samplePatchWidth = 60#80
sqrt_pi = ti.sqrt(ti.pi)

@ti.pyfunc
def radians(degree):
    return degree * ti.pi / 180.

@ti.func
def rotationMatrix(axis, angle):
    axis = ts.normalize(axis)
    s = ts.sin(angle)
    c = ts.cos(angle)
    oc = 1.0 - c
    return ti.Matrix\
    (
    [[oc * axis[0] * axis[0] + c, oc * axis[0] * axis[1] - axis[2] * s, oc * axis[2] * axis[0] + axis[1] * s, 0.0],
     [oc * axis[0] * axis[1] + axis[2] * s, oc * axis[1] * axis[1] + c, oc * axis[1] * axis[2] - axis[0] * s, 0.0],
     [oc * axis[2] * axis[0] - axis[1] * s, oc * axis[1] * axis[2] + axis[0] * s, oc * axis[2] * axis[2] + c, 0.0],
     [0.0, 0.0, 0.0, 1.0]]
    )

@ti.func
def rotate(v, axis, angle):
    m = rotationMatrix(axis, angle)
    V4 = m @ ts.vec4(v, 1.0)
    return tina.V(V4.x, V4.y, V4.z)

@ti.func
def refract(ia, etai, etat):
    cosi = ti.cos(ia)
    sint = (etai / etat) * ti.sqrt(max(0., 1. - cosi*cosi))
    return ti.acos(ti.sqrt(max(0., 1. - sint*sint)))

@ti.func
def fresnelSchlick(cosi, etai, etat):
    if cosi < 0.:
        etai, etat = etat, etai
    kr = 0.
    eta = etai / etat
    f0 = (etai - etat) / (etai + etat)
    f0 *= f0
    sint2_t = eta*eta*max(0., 1. - cosi*cosi)
    if sint2_t > 1.:
        kr = 1.
    else:
        x = 1. - ti.abs(cosi)
        x5 = x*x
        x5 *= x5*x
        kr = f0 + (1. - f0) * x5
    return kr

@ti.func
def fresnelCos(cosi, etai, etat):
    if cosi < 0.:
        etai, etat = etat, etai
    kr = 0.
    eta = etai / etat
    sint2_t = eta*eta * max(0., 1. - cosi*cosi)
    if sint2_t > 1.:
        kr = 1.
    else:
        cost = ti.sqrt(max(1e-3, 1. - sint2_t))
        cosi = ti.abs(cosi)
        Rpar = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost))
        Rper = ((etai * cost) - (etat * cosi)) / ((etai * cost) + (etat * cosi))
        kr = (Rpar * Rpar + Rper * Rper) / 2.0
    return kr

@ti.func
def fresnel(ia, etai, etat):
    cosi = ti.cos(ia)
    return fresnelCos(cosi)

@ti.func
def unitLengthGauss(x, s):
    s = max(1e-3, s)
    result = ti.exp(-(x*x)/(s*s))
    return ti.select(result < 1e-3, 0, result) # Denoising

@ti.func
def unitAreaGauss(x, s):
    result = unitLengthGauss(x, s) / max(1e-3, s*sqrt_pi)
    return ti.select(result < 1e-3, 0, result) # Denoising

@ti.func
def bravais(projectedAngle, n):
    sprj = ti.sin(projectedAngle)
    cprj = ti.cos(projectedAngle)
    return ti.sqrt(n*n - sprj*sprj) / max(1e-3, cprj)

@ti.func
def adjust(a, b, c):
    return (1. - a) * b * c + a * min(b, c)

@ti.data_oriented
class SphericalCoordArgs:
    def __init__(self, **kwawrgs):
        self.n = ti.Vector.field(3, float, ())
        self.b = ti.Vector.field(3, float, ())
        self.t = ti.Vector.field(3, float, ())
        self.i = ti.Vector.field(3, float, ())
        self.r = ti.Vector.field(3, float, ())
        self.ip = ti.Vector.field(3, float, ())
        self.rp = ti.Vector.field(3, float, ())
        self.ips = ti.Vector.field(3, float, ())
        self.rps = ti.Vector.field(3, float, ())

        self.ti = ti.field(float, ())
        self.tr = ti.field(float, ())
        self.th = ti.field(float, ())
        self.td = ti.field(float, ())
        self.pi = ti.field(float, ())
        self.pr = ti.field(float, ())
        self.pd = ti.field(float, ())
        self.psi = ti.field(float, ())
        self.psr = ti.field(float, ())
        self.psd = ti.field(float, ())


@ti.func
def project(args):
    args.ip[None] = ts.normalize(args.n * ts.dot(args.i, args.n) + args.b * ts.dot(args.i, args.b))
    args.rp[None] = ts.normalize(args.n * ts.dot(args.r, args.n) + args.b * ts.dot(args.r, args.b))
    args.ips[None] = ts.normalize(args.n * ts.dot(args.i, args.n) + args.t * ts.dot(args.i, args.t))
    args.rps[None] = ts.normalize(args.n * ts.dot(args.r, args.n) + args.t * ts.dot(args.r, args.t))


@ti.func
def theta(args):
    args.ti[None] = ts.acos(max(0., ts.dot(args.ip, args.i)))
    args.ti[None] *= ts.sign(ts.dot(args.i, args.t))
    args.tr[None] = ts.acos(max(0., ts.dot(args.rp, args.r)))
    args.tr[None] *= ts.sign(ts.dot(args.r, args.t))
    args.th[None] = (args.ti + args.tr) / 2.0
    args.td[None] = (args.ti - args.tr) / 2.0

@ti.func
def phi(args):
    args.pi[None] = ts.acos(max(0., ts.dot(args.ip, args.n)))
    args.pi[None] *= ts.sign(ts.dot(args.ip, args.b))
    args.pr[None] = ts.acos(max(0., ts.dot(args.rp, args.n)))
    args.pr[None] *= ts.sign(ts.dot(args.rp, args.b))
    args.pd[None] = args.pi[None] - args.pr[None]
    
@ti.func
def psi(args):
    args.psi[None] = ts.acos(max(0., ts.dot(args.ips, args.n)))
    args.psi[None] *= ts.sign(ts.dot(args.ips, args.t))
    args.psr[None] = ts.acos(max(0., ts.dot(args.rps, args.n)))
    args.psr[None] *= ts.sign(ts.dot(args.rps, args.t))
    args.psd[None] = args.psi[None] - args.psr[None]

@ti.data_oriented
class MicroParams:
    def __init__(self, **kwargs):
        self.params = kwargs

        self.A = ti.Vector.field(3, float, ())

        self.ni = ti.field(float, ())
        self.nv = ti.field(float, ())
        self.kd = ti.field(float, ())
        self.ys = ti.field(float, ())
        self.yv = ti.field(float, ())

        self.angle = ti.field(float, ())
        self.alpha = ti.field(float, ())
        self.TOF = ti.field(ti.i32, ())
        self.TLEN = ti.field(ti.i32, ())

        self.tan_offsets = ti.field(float, 8)
        self.tan_lens = ti.field(float, 8)

    @ti.func
    def copy_from(self, src):
        self.A[None] = src.A[None]

        self.ni[None] = src.ni[None]
        self.nv[None] = src.nv[None]
        self.kd[None] = src.kd[None]
        self.ys[None] = src.ys[None]
        self.yv[None] = src.yv[None]

        self.angle[None] = src.angle[None]
        self.alpha[None] = src.alpha[None]
        self.TOF[None] = src.TOF[None]
        self.TLEN[None] = self.TLEN[None]

        for i in range(8):
            self.tan_offsets[i] = src.tan_offsets[i]
            self.tan_lens[i] = src.tan_lens[i]

def fill_params_A(params_p, params_v):
    params_p.angle[None] = radians(135.)
    params_p.A[None] = ts.vec3(0.2, 0.8, 1.0) * 0.3
    params_p.ni[None] = 1.0
    params_p.nv[None] = 1.46
    params_p.kd[None] = 0.3
    params_p.ys[None] = radians(12.0)
    params_p.yv[None] = radians(24.0)

    params_p.TOF[None] = 2
    params_p.tan_offsets[0] = radians(-25.0)
    params_p.tan_offsets[1] = radians(25.0)
    params_p.alpha[None] = 0.33

    params_p.TLEN[None] = 1
    params_p.tan_lens[0] = 1.0

    params_v.angle[None] = radians(45.0)
    params_v.A[None] = ts.vec3(0.2, 0.8, 1.0) * 0.6
    params_v.ni[None] = 1.0
    params_v.nv[None] = 1.46
    params_v.kd[None] = 0.3
    params_v.ys[None] = radians(12.0)
    params_v.yv[None] = radians(24.0)

    params_v.TOF[None] = 2
    params_v.tan_offsets[0] = radians(-25.0)
    params_v.tan_offsets[1] = radians(25.0)
    params_v.alpha[None] = 0.33

    params_v.TLEN[None] = 1
    params_v.tan_lens[0] = 1.0
    
def fill_params_B(params_p, params_v):
    params_p.angle[None] = radians(135.)
    params_p.A[None] = ts.vec3(1.0, 0.95, 0.05) * 0.12
    params_p.ni[None] = 1.0
    params_p.nv[None] = 1.345
    params_p.kd[None] = 0.2
    params_p.ys[None] = radians(5.0)
    params_p.yv[None] = radians(10.0)

    params_p.TOF[None] = 4
    params_p.tan_offsets[0] = radians(-35.0)
    params_p.tan_offsets[1] = radians(-35.0)
    params_p.tan_offsets[2] = radians(35.0)
    params_p.tan_offsets[3] = radians(35.0)
    params_p.alpha[None] = 0.75

    params_p.TLEN[None] = 3
    params_p.tan_lens[0] = 1.0
    params_p.tan_lens[1] = 1.0
    params_p.tan_lens[2] = 1.0

    params_v.angle[None] = radians(45.0)
    params_v.A[None] = ts.vec3(1.0, 0.95, 0.05) * 0.16
    params_v.ni[None] = 1.0
    params_v.nv[None] = 1.345
    params_v.kd[None] = 0.3
    params_v.ys[None] = radians(18.0)
    params_v.yv[None] = radians(32.0)

    params_v.TOF[None] = 2
    params_v.tan_offsets[0] = radians(0.0)
    params_v.tan_offsets[1] = radians(0.0)
    params_v.alpha[None] = 0.25

    params_v.TLEN[None] = 1
    params_v.tan_lens[0] = 1.0

def fill_params_C(params_p, params_v):
    params_p.angle[None] = radians(90.0)
    params_p.A[None] = ts.vec3(1.0, 0.37, 0.3) * 0.035
    params_p.ni[None] = 1.0
    params_p.nv[None] = 1.539
    params_p.kd[None] = 0.1
    params_p.ys[None] = radians(2.5)
    params_p.yv[None] = radians(5.0)

    params_p.TOF[None] = 8
    params_p.tan_offsets[0] = radians(32.0)
    params_p.tan_offsets[1] = radians(32.0)
    params_p.tan_offsets[2] = radians(18.0)
    params_p.tan_offsets[3] = radians(0.0)
    params_p.tan_offsets[4] = radians(0.0)
    params_p.tan_offsets[5] = radians(-18.0)
    params_p.tan_offsets[6] = radians(-32.0)
    params_p.tan_offsets[7] = radians(-32.0)
    params_p.alpha[None] = 0.9

    params_p.TLEN[None] = 7
    params_p.tan_lens[0] = 1.33
    params_p.tan_lens[1] = 0.66
    params_p.tan_lens[2] = 2.0
    params_p.tan_lens[3] = 2.0
    params_p.tan_lens[4] = 2.0
    params_p.tan_lens[5] = 0.66
    params_p.tan_lens[6] = 1.33

    params_v.angle[None] = radians(0.0)
    params_v.A[None] = ts.vec3(1.0, 0.37, 0.3) * 0.2
    params_v.ni[None] = 1.0
    params_v.nv[None] = 1.539
    params_v.kd[None] = 0.7
    params_v.ys[None] = radians(30.0)
    params_v.yv[None] = radians(60.0)

    params_v.TOF[None] = 2
    params_v.tan_offsets[0] = radians(0.0)
    params_v.tan_offsets[1] = radians(0.0)
    params_v.alpha[None] = 0.1

    params_v.TLEN[None] = 1
    params_v.tan_lens[0] = 1.0

def fill_params_D(params_p, params_v):
    params_p.angle[None] = radians(135.0)
    params_p.A[None] = ts.vec3(1.0, 0.37, 0.3) * 0.035
    params_p.ni[None] = 1.0
    params_p.nv[None] = 1.539
    params_p.kd[None] = 0.1
    params_p.ys[None] = radians(2.5)
    params_p.yv[None] = radians(5.0)

    params_p.TOF[None] = 8
    params_p.tan_offsets[0] = radians(-30.0)
    params_p.tan_offsets[1] = radians(-30.0)
    params_p.tan_offsets[2] = radians(30.0)
    params_p.tan_offsets[3] = radians(30.0)
    params_p.tan_offsets[4] = radians(-5.0)
    params_p.tan_offsets[5] = radians(-5.0)
    params_p.tan_offsets[6] = radians(5.0)
    params_p.tan_offsets[7] = radians(5.0)
    params_p.alpha[None] = 0.67

    params_p.TLEN[None] = 7
    params_p.tan_lens[0] = 1.33
    params_p.tan_lens[1] = 1.33
    params_p.tan_lens[2] = 1.33
    params_p.tan_lens[3] = 0.0
    params_p.tan_lens[4] = 0.67
    params_p.tan_lens[5] = 0.67
    params_p.tan_lens[6] = 0.67

    params_v.angle[None] = radians(45.0)
    params_v.A[None] = ts.vec3(1.0, 0.37, 0.3) * 0.2
    params_v.ni[None] = 1.0
    params_v.nv[None] = 1.46
    params_v.kd[None] = 0.7
    params_v.ys[None] = radians(30.0)
    params_v.yv[None] = radians(60.0)

    params_v.TOF[None] = 2
    params_v.tan_offsets[0] = radians(0.0)
    params_v.tan_offsets[1] = radians(0.0)
    params_v.alpha[None] = 0.33

    params_v.TLEN[None] = 1
    params_v.tan_lens[0] = 3.0

def fill_params_E(params_p, params_v):
    params_p.angle[None] = radians(135.0)
    params_p.A[None] = ts.vec3(0.1, 1.0, 0.4) * 0.2
    params_p.ni[None] = 1.0
    params_p.nv[None] = 1.345
    params_p.kd[None] = 0.1
    params_p.ys[None] = radians(4.0)
    params_p.yv[None] = radians(8.0)

    params_p.TOF[None] = 4
    params_p.tan_offsets[0] = radians(-25.0)
    params_p.tan_offsets[1] = radians(-25.0)
    params_p.tan_offsets[2] = radians(25.0)
    params_p.tan_offsets[3] = radians(25.0)
    params_p.alpha[None] = 0.86

    params_p.TLEN[None] = 3
    params_p.tan_lens[0] = 1.33
    params_p.tan_lens[1] = 2.67
    params_p.tan_lens[2] = 1.33

    params_v.angle[None] = radians(45.0)
    params_v.A[None] = ts.vec3(1.0, 0.0, 0.1) * 0.6
    params_v.ni[None] = 1.0
    params_v.nv[None] = 1.345
    params_v.kd[None] = 0.1
    params_v.ys[None] = radians(5.0)
    params_v.yv[None] = radians(10.0)

    params_v.TOF[None] = 2
    params_v.tan_offsets[0] = radians(0.0)
    params_v.tan_offsets[1] = radians(0.0)
    params_v.alpha[None] = 0.14

    params_v.TLEN[None] = 2
    params_v.tan_lens[0] = 3.0

def fill_params_F(params_p, params_v):
    params_p.angle[None] = radians(135.0)
    params_p.A[None] = ts.vec3(0.75, 0.02, 0.0) * 0.3
    params_p.ni[None] = 1.0
    params_p.nv[None] = 1.46
    params_p.kd[None] = 0.1
    params_p.ys[None] = radians(6.0)
    params_p.yv[None] = radians(12.0)

    params_p.TOF[None] = 2
    params_p.tan_offsets[0] = radians(-90.0)
    params_p.tan_offsets[1] = radians(-50.0)
    params_p.alpha[None] = 0.5

    params_p.TLEN[None] = 2
    params_p.tan_lens[0] = 1.0

    params_v.angle[None] = radians(45.0)
    params_v.A[None] = ts.vec3(0.55, 0.02, 0.0) * 0.3
    params_v.ni[None] = 1.0
    params_v.nv[None] = 1.46
    params_v.kd[None] = 0.1
    params_v.ys[None] = radians(6.0)
    params_v.yv[None] = radians(12.0)

    params_v.TOF[None] = 4
    params_v.tan_offsets[0] = radians(-90.0)
    params_v.tan_offsets[1] = radians(-55.0)
    params_v.tan_offsets[2] = radians(55.0)
    params_v.tan_offsets[3] = radians(90.0)
    params_v.alpha[None] = 0.5

    params_v.TLEN[None] = 3
    params_v.tan_lens[0] = 0.5
    params_v.tan_lens[1] = 0.0
    params_v.tan_lens[2] = 0.5

@ti.data_oriented
class ParamPrefab:
    def __init__(self):
        self.prefab_paramsV = np.array([MicroParams() for _ in range(6)])
        self.prefab_paramsP = np.array([MicroParams() for _ in range(6)])
        self.prefab_tanOffsetsP = np.array([ti.field(float, 8) for _ in range(6)])
        self.prefab_tanLensP = np.array([ti.field(float, 8) for _ in range(6)])
        self.prefab_tanOffsetsV = np.array([ti.field(float, 8) for _ in range(6)])
        self.prefab_tanLensV = np.array([ti.field(float, 8) for _ in range(6)])
        # self.prefab_TOFP = np.array([ti.field(ti.i32, ()) for _ in range(6)])
        # self.prefab_TOFV = np.array([ti.field(ti.i32, ()) for _ in range(6)])
        # self.prefab_TLENP = np.array([ti.field(ti.i32, ()) for _ in range(6)])
        # self.prefab_TLENV = np.array([ti.field(ti.i32, ()) for _ in range(6)])
        self.prefab_a1 = np.array([ti.field(float, ()) for _ in range(6)])
        self.prefab_a2 = np.array([ti.field(float, ()) for _ in range(6)])

        @ti.materialize_callback
        def init_prefabs():
            fill_param_func_dict = {
                0:fill_params_A,
                1:fill_params_B,
                2:fill_params_C,
                3:fill_params_D,
                4:fill_params_E,
                5:fill_params_F,
            }
            null_vec = ts.vec(0., 1., 0.)

            @ti.kernel
            def aux():
                self.tmp_func(
                    self.tmp_paramsP, 
                    self.tmp_tanOffsetsP, 
                    self.tmp_tanLensP,
                    self.tmp_a1,
                    self.tmp_paramsV, 
                    self.tmp_tanOffsetsV, 
                    self.tmp_tanLensV,
                    self.tmp_a2,
                    null_vec, null_vec, null_vec, null_vec, null_vec)

            for i in fill_param_func_dict:
                self.tmp_func = fill_param_func_dict[i]
                self.tmp_paramsP = self.prefab_paramsP[i]
                self.tmp_tanOffsetsP = self.prefab_tanOffsetsP[i]
                self.tmp_tanLensP = self.prefab_tanLensP[i]
                self.tmp_a1 = self.prefab_a1[i]
                self.tmp_paramsV = self.prefab_paramsV[i]
                self.tmp_tanOffsetsV = self.prefab_tanOffsetsV[i]
                self.tmp_tanLensV = self.prefab_tanLensV[i]
                self.tmp_a2 = self.prefab_a2[i]
                aux()

    @ti.func
    def fill_param(self, index,  
                   paramsP, tanOffsetsP, tanLensP, a1,
                   paramsV, tanOffsetsV, tanLensV, a2,
                   n, t, b, i, r):
        paramsP.copy_from(self.prefab_paramsP[index])
        paramsP.t[None] = rotate(t, n, self.prefab_paramsP[index].angle[None])
        paramsP.b[None] = rotate(b, n, self.prefab_paramsP[index].angle[None])

        paramsV.copy_from(self.prefab_paramsV[index])
        paramsV.t[None] = rotate(t, n, self.prefab_paramsV[index].angle[None])
        paramsV.b[None] = rotate(b, n, self.prefab_paramsV[index].angle[None])

        for _ in range(8):
            tanOffsetsP[_] = self.prefab_tanOffsetsP[index][_]
            tanLensP[_] = self.prefab_tanLensP[index][_]

            tanOffsetsV[_] = self.prefab_tanOffsetsV[index][_]
            tanLensV[_] = self.prefab_tanLensV[index][_]

        a1[None] = self.prefab_a1[index][None]
        a2[None] = self.prefab_a2[index][None]

unitHeightSTD = 25.

#fresnelUsing = fresnelCos
fresnelUsing = fresnelSchlick

@ti.func
def toSpherical(n):
    r = ts.length(n)
    azimuth = ts.atan(n.x, n.z)
    zenith = ts.acos(n.y / r)
    return tina.V(azimuth, zenith)

@ti.data_oriented
class SadegiClothMaterial(tina.IMaterial):
    arguments = ['index']
    defaults = [2]

    def __init__(self, 
                 param_index=2, 
                 **kwargs):
        #self.index = tina.Param(dtype=int, initial=param_index)
        #kwargs['index'] = self.index
        super(tina.IMaterial, self).__init__(**kwargs)
        self.fill_param_func_dict = {
            0:fill_params_A,
            1:fill_params_B,
            2:fill_params_C,
            3:fill_params_D,
            4:fill_params_E,
            5:fill_params_F,
        }
        #self.fill_param(param_index)

        self.init_taichi_values()
        
        @ti.materialize_callback
        def init_param():
            self.init_fill_param(param_index)

    def init_fill_param(self, index):
        self.fill_param_func_dict.get(index, fill_params_A)(self.params_p, self.params_v)

    def init_taichi_values(self):
        self.param_index = ti.field(ti.i32, ())

        self.params_p = MicroParams()
        self.params_v = MicroParams()
        self.spcoord_args = SphericalCoordArgs()
    
    @ti.func
    def brdf(self, nrm, idir, odir):
        #print(nrm, idir, odir)#TEST
        up=tina.V(1., 0., 0.)
        up2=tina.V(0., 0., 1.)
        bitan = nrm.cross(up)
        bitan = ti.select(bitan.norm() < 1e-3,
                          nrm.cross(up2).normalized(),
                          bitan.normalized())
        tan = bitan.cross(nrm)

        val = self.appearance(self.params_p, self.params_v, self.spcoord_args,
                              nrm, tan, bitan, idir, odir)

        return val
        
    # @classmethod
    # def cook_for_ibl(cls, tab, precision):
    #     env = tab['env']
    #     ibl = tina.Skybox(env.resolution // 6)
    #     denoise = tina.Denoise(ibl.shape)
    #     nsamples = 128 * precision

    #     @ti.kernel
    #     def bake():
    #         # https://zhuanlan.zhihu.com/p/261005894
    #         for I in ti.grouped(ibl.img):
    #             dir = ibl.unmapcoor(I)
    #             res, dem = tina.V(0., 0., 0.), 0.
    #             for s in range(nsamples):
    #                 u, v = ti.random(), ti.random()
    #                 odir = tina.tangentspace(dir) @ tina.spherical(u, v)
    #                 wei = env.sample(odir)
    #                 res += wei * u
    #                 dem += u
    #             ibl.img[I] = res / dem

    #     @ti.materialize_callback
    #     def init_ibl():
    #         print(f'[Tina] Baking IBL map ({"x".join(map(str, ibl.shape))} {nsamples} spp) for Lambert...')
    #         bake()
    #         print('[Tina] Denoising IBL map with KNN for Lambert...')
    #         denoise.src.copy_from(ibl.img)
    #         denoise.knn()
    #         ibl.img.copy_from(denoise.dst)
    #         print('[Tina] Baking IBL map for Lambert done')

    #     tab['diff'] = ibl

    # @ti.func
    # def sample_ibl(self, tab, idir, nrm):
    #     return tab['diff'].sample(nrm)

        
    @ti.func
    def appearance(self, params_p, params_v, spcoord_args,
                   nrm, tan, bitan, idir, odir):
        sump = ts.vec3(0.0, 0.0, 0.0)
        Q = 0.
        trw = 0.
        
        # Create sampels per tangent curve.
        curveLen = 0.
        tlen = params_p.TLEN[None]
        for i in range(tlen):#(int i = 0; i < TLENP; i++)
            curveLen += params_p.tan_lens[i]
        # for i in range(TLENV):#(int i = 0; i < TLENV; i++)
        #     curveLen += params_v.tan_lens[i]
        
        k = float(samplePatchWidth) / curveLen # patch num of this pixel
        counter = 0
        tlen = params_p.TLEN[None]
        
        tan_p = rotate(tan, nrm, params_p.angle[None])
        bitan_p = rotate(bitan, nrm, params_p.angle[None])
        
        for i in range(tlen):#(int i = 0; i < TLENP; i++)
            if params_p.tan_lens[i] != 0.0:
                c = max(1, int(ts.round(k * params_p.tan_lens[i])))
                inv_c = ti.select(c == 1, 0.5, 1. / float(c - 1))
                offsetsDiff = (params_p.tan_offsets[i + 1] - params_p.tan_offsets[i])

                # Sum micro cylinder result per sample.
                # c is the occurence num of this patch?
                for j in range(c):#(int j = 0; j < c; j++)
                    # params_p.tan_offsets[i] + (t[i+1]-t[i])*(j/c)
                    offAngle = params_p.tan_offsets[i] + offsetsDiff * float(j) * inv_c
                    local_nrm = rotate(nrm, bitan_p, offAngle)
                    local_tan = rotate(tan_p, bitan_p, offAngle)
                    microP, rw = self.micro(params_p, spcoord_args, local_nrm, local_tan, bitan_p, idir, odir)
                    sump += microP
                    trw += rw
                    counter += 1

        # Average it.
        sump /= float(counter)
        trw /= float(counter)
        Q += trw * params_p.alpha[None]

        counter = 0

        # Do same for second yarn.
        sumv = ts.vec3(0.0, 0.0, 0.0)
        trw = 0.0
        tlen = params_v.TLEN[None]
        
        tan_v = rotate(tan, nrm, params_v.angle[None])
        bitan_v = rotate(bitan, nrm, params_v.angle[None])

        for i in range(tlen):#(int i = 0; i < TLENV; i++)
            if params_v.tan_lens[i] != 0.0:
                c = max(1, int(ts.round(k * params_v.tan_lens[i])))
                inv_c = ti.select(c == 1, 0.5, 1. / float(c - 1))
                offsetsDiff = (params_v.tan_offsets[i + 1] - params_v.tan_offsets[i])

                for j in range(c):#(int j = 0; j < c; j++)
                    offAngle = params_v.tan_offsets[i] + offsetsDiff * float(j) * inv_c
                    local_nrm = rotate(nrm, bitan_v, offAngle)
                    local_tan = rotate(tan_v, bitan_v, offAngle)
                    microV, rw = self.micro(params_v, spcoord_args, local_nrm, local_tan, bitan_v, idir, odir)
                    sumv += microV
                    trw += rw
                    counter += 1
        sumv /= float(counter)
        trw /= float(counter)
        Q += trw * params_v.alpha[None]

        Q += (1. - params_p.alpha[None] - params_v.alpha[None]) * ts.dot(nrm, odir)

        return (params_p.alpha[None] * sump + params_v.alpha[None] * sumv) / max(1e-3, Q)
        
    @ti.func
    def micro(self, params, args, nrm, tan, bitan, idir, odir):
        args.n[None] = nrm
        args.b[None] = bitan
        args.t[None] = tan
        args.i[None] = idir
        args.r[None] = odir

        project(args)
        theta(args)
        phi(args)
        psi(args)
        
        # params.nv = bravais(args.ti, params.nv)
        cos_td = max(0., ts.cos(args.td[None]))
        cos_half_pd = max(0., ts.cos(args.pd[None] / 2.0))
        cos_ti = max(0., ts.cos(args.ti[None]))
        cos_ti_s = max(1e-3, cos_ti)
        cos_tr_s = max(1e-3, ts.cos(args.tr[None]))
        cos_td_s = max(1e-3, cos_td)
        cos_pi = max(0., ts.cos(args.pi[None]))
        cos_pr = max(0., ts.cos(args.pr[None]))
        cos_psi = max(0., ts.cos(args.psi[None]))
        cos_psr = max(0., ts.cos(args.psr[None]))

        ###ia = ts.acos(cos_td * cos_half_pd)

        #Fri_exact = fresnelUsing(cos_td*cos_half_pd, params.ni[None], params.nv[None])###fresnel(ia, params.ni[None], params.nv[None])
        Fri = fresnelUsing(ts.dot(args.i[None], args.n[None]), params.ni[None], params.nv[None])
        ###fresnel(ts.acos(ts.dot(args.i[None], args.n[None])), params.ni[None], params.nv[None])
        
        Fti = 1. - Fri
        Ftr_prime = 1. - fresnelUsing(ts.dot(-args.r[None], args.n[None]), params.nv[None], params.ni[None])
        #Ftr = 1. - fresnelUsing(ts.dot(args.r[None], args.n[None]), params.nv[None], params.ni[None])

        frs = Fri * cos_half_pd * unitAreaGauss(args.th[None], params.ys[None])
        #frs = Fri_exact * cos_half_pd * unitAreaGauss(args.th[None], params.ys[None])

        F = Fti * Ftr_prime # F is not relevant of black points
        #F = Fti * Ftr

        frv = F * (((1.0 - params.kd[None]) * unitAreaGauss(args.th[None], params.yv[None]) + params.kd[None]) / (cos_ti_s + cos_tr_s)) * params.A[None]
        
        #res = cos_ti * (ts.vec3(frs) + frv) / (cos_td_s * cos_td_s)
        res = (ts.vec3(frs) + frv) / (cos_td_s * cos_td_s)

        sm = adjust(unitLengthGauss(args.pd[None], radians(unitHeightSTD)), cos_pi, cos_pr) # Shadowing & masking

        # args.b = params.t * rw
        # project(args)
        # phi(args)

        #rw = 1.0 
        
        rw = adjust(unitLengthGauss(args.psd[None], radians(unitHeightSTD)), cos_psi, cos_psr) # Reweighting

        return rw * sm * res, rw

        
