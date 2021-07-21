import taichi as ti
import taichi_glsl as ts
import numpy as np
import tina
import MyImpl.MyScene as MyScene
import MyImpl.MyMaterial as MyMaterial

ti.init(arch=ti.gpu, advanced_optimization=True, device_memory_GB=2.0)

### Parameters

N = 128#128
NN = N, N
W = 1
rest_length = W / N
mass0 = 2.#0.5#1. / (N*N)#0.1
gravity = 0.1#0.1#0.5#1.#0.5
stiffness = 1000.#1000.#3600.#2400#1600# * N / W
ball_radius = 0.4
damping = 2#2
steps = 30#30
isteps = 10
opt_steps = 30
dt = 5e-4#1e-3

### Physics

Z3 = ti.Matrix([[0.,0.,0.],
                [0.,0.,0.],
                [0.,0.,0.]])
I3 = ti.Matrix([[1.,0.,0.],
                [0.,1.,0.],
                [0.,0.,1.]])

x = ti.Vector.field(3, float, NN)
v = ti.Vector.field(3, float, NN)
#f = ti.Vector.field(3, float, NN)

mass = ti.field(float, NN)
inv_mass = ti.field(float, NN)

ball_pos = ti.Vector.field(3, float, ())
ball_vel = ti.Vector.field(3, float, ())


links = [(-1, 0), (1, 0), (0, -1), (0, 1), \
         (-1, -1), (1, -1), (-1, 1), (1, 1), \
         (-2, 0), (2, 0), (0, -2), (0, 2), ]
links = [ti.Vector(_) for _ in links]

n_links = len(links)
M = N * N
# S = M * n_links

# y = ti.Vector.field(3, float, NN)
# L_ti = ti.field(float, (3*N, 3*N, 3*N, 3*N))
# A_ti = ti.field(float, (S, N, N))
# S_ti = ti.field(float, (S, S))
# D_ti = ti.Vector.field(3, float, S)

# M_np = None#np.eye(3*N*N) * mass0
# L_np = None#np.eye(3*N*N)
# J_np = None
# D_np = None
# y_np = np.zeros((3*M,), dtype=float)
# x_np = np.zeros((3*M,), dtype=float)
# G_np = None

# InertMat_np = None
# InertMat_Cholesky_np = None

# '''Start Simulation Implementation'''

# '''Start Fast'''
# def to_numpy_span(tiarray, dim=0):
#     nparray = tiarray.to_numpy()
#     shape = nparray.shape[0:dim] + (-1,)
#     return nparray.reshape(shape)

# @ti.kernel
# def x_to_taichi(x_np : ti.ext_arr()):
#     for I in ti.grouped(x):
#         index = I[0] * N + I[1]
#         x[I][0] = x_np[index * 3 + 0]
#         x[I][1] = x_np[index * 3 + 1]
#         x[I][2] = x_np[index * 3 + 2]
        
# @ti.kernel
# def G_to_numpy(G_np : ti.ext_arr()):
#     for I in ti.grouped(x):
#         index = I[0] * N + I[1]
#         G_np[index * 3 + 0] = 0.
#         G_np[index * 3 + 1] = -gravity * mass[I]
#         G_np[index * 3 + 2] = 0.
        
# @ti.kernel
# def x_to_numpy(x_np : ti.ext_arr()):
#     for I in ti.grouped(x):
#         index = I[0] * N + I[1]
#         x_np[index * 3 + 0] = x[I][0]
#         x_np[index * 3 + 1] = x[I][1]
#         x_np[index * 3 + 2] = x[I][2]
        
# @ti.kernel
# def init_fast_taichi():
#     for I in ti.grouped(x):
#         index = 0
#         product = I[0] * I[1]
#         n_links = len(links)
#         for D in ti.static(links):
#             J = min(max(I + D, 0), ti.Vector(NN) - 1)
#             A_ti[product*n_links + index, I[0], I[1]] = ti.select(inv_mass[I] == 0., 0., 1.)
#             A_ti[product*n_links + index, J[0], J[1]] = ti.select(inv_mass[J] == 0., 
#                                                           0., 
#                                                           ti.select(A_ti[index, I[0], I[1]] == 0., 
#                                                                     -1., 
#                                                                     0.))
#             index += 1
#     for i in range(S):
#         S_ti[i, i] = 1.

# def init_fast_numpy(dt):
#     global M_np
#     global L_np
#     global J_np
#     global InertMat_np
#     global InertMat_Cholesky_np
#     global G_np

#     M_np = np.eye(3*M) * mass0

#     I3_np = I3.to_numpy()
#     A_np = to_numpy_span(A_ti, dim=1)
#     S_np = to_numpy_span(S_ti, dim=1)

#     print(A_np.shape, S_np.shape)

#     L_pre = np.zeros((M, M), dtype=float)
#     J_pre = np.zeros((M, S), dtype=float)
#     for s in range(A_np.shape[0]):
#         A_np_s = A_np[s]
#         L_pre += stiffness * (np.outer(A_np_s, A_np_s))
#         J_pre += stiffness * (np.outer(A_np_s, S_np[s]))
#     L_np = np.kron(L_pre, I3_np)
#     J_np = np.kron(J_pre, I3_np)
    
#     InertMat_np = M_np + (dt*dt)*L_np
#     InertMat_Cholesky_np = np.linalg.cholesky(InertMat_np)
    
#     G_np = np.zeros((3*M,), dtype = float)
#     G_to_numpy(G_np)
#     x_to_numpy(x_np)

# def solve_cholesky(a, b):
#     import scipy.linalg as alg
#     y = alg.solve_triangular(a, b, trans=0)
#     #x = alg.solve_triangular(a, y, trans=1)
#     x = alg.solve_triangular(a.transpose(), y, trans=0)
#     return x

# @ti.kernel
# def calc_y(dt : float):
#     for I in ti.grouped(y):
#         y[I] = x[I] + dt*v[I]
#         x[I] = y[I]
        
# @ti.kernel
# def calc_v(dt : float):
#     for I in ti.grouped(v):
#         v[I] = (x[I] - y[I]) / dt + v[I]
#         v[I] *= ti.exp(-damping * dt)#Damping
#     move_ball(dt)

# def substep_global(dt):
#     global x_np
#     global D_np
#     global y_np
#     y_np = to_numpy_span(y, dim=0)
#     D_np = to_numpy_span(D_ti, dim=0)
#     b = (dt*dt)*np.matmul(J_np, D_np) + np.matmul(M_np, y_np + G_np)
#     x_np = solve_cholesky(InertMat_Cholesky_np, b)
#     x_to_taichi(x_np)
    

# @ti.kernel
# def substep_local():
#     for I in ti.grouped(x):
#         index = 0
#         product = I[0] * I[1]
#         for D in ti.static(links):
#             J = min(max(I + D, 0), ti.Vector(NN) - 1)
#             disp = x[I] - x[J]
#             disp_length = disp.norm()
#             length = rest_length * float(D).norm()
#             D_ti[product + index] = ti.select(disp_length >= 1e-5, 
#                                               disp * (length / disp_length),
#                                               ts.vec3(0.))

# def substep_fast(dt):
#     global y_np
#     calc_y(dt)
#     y_np = to_numpy_span(y)
#     for _ in range(opt_steps):
#         substep_local()
#         substep_global(dt)
#     calc_v(dt)

# '''End Fast'''

@ti.func
def move_ball(dt):
    ball_pos[None] += ball_vel[None] * dt

'''Start Explicit'''

@ti.kernel
def init_explicit():
    ball_pos[None] = ti.Vector([0.0, +0.0, 0.0])
    for i in ti.grouped(x):
        fix = (i[0] == 0 and i[1] == 0 or i[0] == N - 1 and i[1] == 0)
        #fix = (i[1] == 0)
        mass[i] = ti.select(fix, 0., mass0)
        inv_mass[i] = ti.select(fix, 0., 1. / mass[i])
        m, n = (i + 0.5) * rest_length - 0.5
        x[i] = ti.Vector([m, 0.6, n])
        v[i] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def substep_explicit(dt : ti.f32):
    move_ball(dt)

    for I in ti.grouped(x):
        acc = x[I] * 0
        for d in ti.static(links):
            J0 = I + d
            J = min(max(J0, 0), ti.Vector(NN) - 1)
            if all(J0 == J):
                disp = x[J] - x[I]
                length = rest_length * float(d).norm()
                #acc += disp * (disp.norm() - length) / length**2 * inv_mass[i]
                acc += disp * (disp.norm() - length) / length * inv_mass[I]
        v[I] += stiffness * acc * dt

    for I in ti.grouped(x):
        v[I].y -= ti.select(inv_mass[I] > 0., gravity * dt, 0.)
        dp = x[I] - ball_pos[None]
        dp2 = dp.norm_sqr()
        if dp2 <= ball_radius**2 + 1e-2:
            # a fun execise left for you: add angular velocity to the ball?
            # it should drag the cloth around with a specific friction rate.
            dv = v[I] - ball_vel[None]
            NoV = dv.dot(dp)
            if NoV < 0:
                v[I] -= NoV * dp / (dp2-1e-2)

    for I in ti.grouped(x):
        v[I] *= ti.exp(-damping * dt)
        x[I] += dt * v[I]
        
        # dp = x[i] - ball_pos[None]
        # dp2 = dp.norm_sqr()
        # if dp2 <= ball_radius**2 + 1e-2:
        #     new_tmp_x = ball_pos[None] + dp.normalized()*(ball_radius+1e-1)
        #     v[i] += (new_tmp_x - x[i]) / dt
        #     x[i] = new_tmp_x
'''End Explicit'''

'''End Simulation Implementation'''

### Rendering GUI

# Hint: remove ibl=True if you find it compiles too slow...
scene = MyScene.MyRasterScene((1024, 768), smoothing=True, fxaa=False, ibl=False)
#tina.Scene((1024, 768), smoothing=True, texturing=True, ibl=True)
#scene = MyScene.MyPathTracingScene((1024, 768), smoothing=True, texturing=True, ibl=True)

mesh = tina.MeshNoCulling(tina.MeshGrid((N, N)))
ball = tina.MeshTransform(tina.PrimitiveMesh.sphere())


cloth_Sadegi = MyMaterial.SadegiClothMaterial(0)
    #tina.PBR(basecolor=tina.Texture('assets/cloth.jpg'))
    #tina.PBR(basecolor=tina.ChessboardTexture(size=0.2))
metal = tina.PBR(basecolor=[1.0, 0.9, 0.8], metallic=0.8, roughness=0.4)

scene.add_object(mesh, cloth_Sadegi)
scene.add_object(ball, metal)


if not isinstance(scene, tina.PTScene):
    scene._ensure_material_shader(cloth_Sadegi)

is_opt_method = False
simulating = True

def modify_patch_width(delta):
    MyMaterial.samplePatchWidth += delta
    MyMaterial.samplePatchWidth = max(1, MyMaterial.samplePatchWidth)
    print('samplePatchWidth =', MyMaterial.samplePatchWidth)

def flip_is_opt_method(e, c):
    global is_opt_method
    is_opt_method = not is_opt_method
    print('opt_method' if is_opt_method else 'explicit')

def flip_is_simulating(e, c):
    global simulating
    simulating = not simulating
    print('simulating' if simulating else 'not simulating')
    
def set_material(index, sadegi):
    if sadegi:
        print('Sadegi', index)
        cloth_Sadegi.init_fill_param(index)
    else:
        pass

extra_input_events = \
{
    (ti.GUI.RELEASE, '0'):flip_is_opt_method,
    (ti.GUI.RELEASE, ti.GUI.SPACE):flip_is_simulating,
}

extra_pressed_events = \
{
    # ti.GUI.UP:lambda : modify_patch_width(1),
    # ti.GUI.DOWN:lambda : modify_patch_width(-1),
}

class AuxiliaryFunc:
    def __init__(self, index):
        self.index = index
    
    def __call__(self, e, c):
        ctrl = c(ti.GUI.CTRL)
        set_material(self.index, True)


for i in range(6):
    key = ascii(i + 1)
    extra_input_events[ti.GUI.RELEASE, key] = AuxiliaryFunc(i)

gui = ti.GUI('Mass Spring', scene.res, fast_gui=not isinstance(scene, tina.PTScene))
#gui = ti.GUI('Mass Spring', scene.res, fast_gui=False)
#gui = ti.GUI('Mass Spring', scene.res)

scene.init_control(gui,
        center=[0.0, 0.0, 0.0],
        theta=np.pi / 2 - np.radians(30),
        radius=1.5,
        input_events=extra_input_events,
        pressed_events=extra_pressed_events)

@ti.kernel
def add():
    ball_pos[None] += ball_vel[None] * dt

@ti.kernel
def update_normal():
    mesh.pre_compute()

init_explicit()
#init_fast_taichi()
#init_fast_numpy(dt*steps)

# for i in range(max(len(cloth_Sadegi), len(cloth_CookTolerance))):
#     set_material(i, False)
#     set_material(i, True)
# scene.render()
ball_vel_factor = 0.5
print('[Hint] Press ASWDEQ to move the ball, R to reset')
while gui.running:
    scene.input(gui)

    if simulating:
        for i in range(steps):
            substep_explicit(dt)
        #substep_fast(dt*steps)

    if gui.is_pressed('r'):
        init_explicit()
        #x_to_numpy(x_np)
        # init_fast_taichi()
        # init_fast_numpy()

    if gui.is_pressed('w'):
        ball_vel[None] = tina.V(0, 0, -1) * ball_vel_factor
    elif gui.is_pressed('s'):
        ball_vel[None] = tina.V(0, 0, +1) * ball_vel_factor
    elif gui.is_pressed('e'):
        ball_vel[None] = tina.V(0, +1, 0) * ball_vel_factor
    elif gui.is_pressed('q'):
        ball_vel[None] = tina.V(0, -1, 0) * ball_vel_factor
    elif gui.is_pressed('a'):
        ball_vel[None] = tina.V(-1, 0, 0) * ball_vel_factor
    elif gui.is_pressed('d'):
        ball_vel[None] = tina.V(+1, 0, 0) * ball_vel_factor
    else:
        ball_vel[None] = tina.V(0, 0, 0)

    mesh.pos.copy_from(x)
    update_normal()
    ball.set_transform(tina.translate(ball_pos[None].value) @ tina.scale(ball_radius))

    if isinstance(scene, tina.PTScene):
        scene.render(nsteps=16)
    else:
        scene.render()
    gui.set_image(scene.img)
    gui.show()
