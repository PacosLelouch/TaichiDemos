import taichi as ti
import taichi_glsl as ts
import numpy as np
import tina
import MyImpl.MyScene as MyScene
import MyImpl.MyMaterial as MyMaterial

ti.init(arch=ti.gpu, advanced_optimization=True, device_memory_GB=2.0)

### Parameters

N = 128
NN = N, N
W = 1
L = W / N
mass0 = 0.2
gravity = 1.#0.5
stiffness = 3200#1600# * N / W
ball_radius = 0.4
damping = 2
idamping = 0#20
collision_ik = 0#1
factor = 3#3
steps = 30
isteps = 10
jacob_steps = 20
dt = 1e-3#5e-4
idt = 1e-3 * (steps / isteps)#5e-4 * (steps / isteps)

### Physics

x = ti.Vector.field(3, float, NN)
v = ti.Vector.field(3, float, NN)
f = ti.Vector.field(3, float, NN)

mass = ti.field(float, NN)
inv_mass = ti.field(float, NN)
Zero = ti.Matrix([[0.,0.,0.],
                  [0.,0.,0.],
                  [0.,0.,0.]])
Identity = ti.Matrix([[1.,0.,0.],
                      [0.,1.,0.],
                      [0.,0.,1.]])
fParXMat = ti.Matrix.field(3, 3, float, NN)
fParVMat = ti.Matrix.field(3, 3, float, NN)
AMat = ti.Matrix.field(3, 3, float, NN)
BVec = ti.Vector.field(3, float, NN)
v_next = ti.Vector.field(3, float, NN)


ball_pos = ti.Vector.field(3, float, ())
ball_vel = ti.Vector.field(3, float, ())

@ti.kernel
def init():
    ball_pos[None] = ti.Vector([0.0, +0.0, 0.0])
    for i in ti.grouped(x):
        fix = (i[0] == 0 and i[1] == 0 or i[0] == N - 1 and i[1] == 0)
        mass[i] = ti.select(fix, 0., mass0)
        inv_mass[i] = ti.select(fix, 0., 1. / mass[i])
        m, n = (i + 0.5) * L - 0.5
        x[i] = ti.Vector([m, 0.6, n])
        v[i] = ti.Vector([0.0, 0.0, 0.0])


links = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]
links = [ti.Vector(_) for _ in links]
'''Start Explicit'''
@ti.func
def apply_gravity(i):
    v[i].y -= ti.select(inv_mass[i] > 0., gravity * dt, 0.)
    dp = x[i] - ball_pos[None]
    dp2 = dp.norm_sqr()
    if dp2 <= ball_radius**2 + 1e-3:
        # a fun execise left for you: add angular velocity to the ball?
        # it should drag the cloth around with a specific friction rate.
        dv = v[i] - ball_vel[None]
        NoV = dv.dot(dp)
        if NoV < 0:
            v[i] -= NoV * dp / dp2
            
@ti.func
def apply_damping(i):
    v[i] += cal_damp_force(i, damping) * inv_mass[i] * dt
    #v[i] *= ti.exp(-damping * dt)

@ti.kernel
def substep_explicit():
    ball_pos[None] += ball_vel[None] * dt

    for i in ti.grouped(x):
        acc = x[i] * 0
        for d in ti.static(links):
            disp = x[min(max(i + d, 0), ti.Vector(NN) - 1)] - x[i]
            length = L * float(d).norm()
            #acc += disp * (disp.norm() - length) / length**2 * inv_mass[i]
            acc += disp * (disp.norm() - length) / length * inv_mass[i]
        v[i] += stiffness * acc * dt

    for i in ti.grouped(x):
        apply_gravity(i)

    for i in ti.grouped(x):
        apply_damping(i)
        x[i] += dt * v[i]
'''End Explicit'''

'''Start Implicit'''
@ti.func
def cal_elastic_force(taridx, srcidx, k, rest_length):
    tarx = x[taridx]
    srcx = x[srcidx]
    cur_diff = tarx - srcx
    cur_length = cur_diff.norm() #ti.sqrt(cur_diff.dot(cur_diff))
    direction = cur_diff / cur_length
    deform_value = cur_length - rest_length
    elastic = -k * deform_value * direction
    return elastic

@ti.func
def cal_collision_force(i):
    force = tina.V(0., 0., 0.)
    next_ball_pos = ball_pos[None] + ball_vel[None]*idt
    dp = x[i] - next_ball_pos
    dp2 = dp.norm_sqr()
    if dp2 <= ball_radius**2 + 1e-3:
        # a fun execise left for you: add angular velocity to the ball?
        # it should drag the cloth around with a specific friction rate.
        dv = v[i] - ball_vel[None]
        NoV = dv.dot(dp)
        if NoV < 0:
            force = -NoV * dp / dp2 * mass[i] / idt
    return force

@ti.func
def cal_damp_force(taridx, kd):
    damp = -kd * v[taridx]# * v[taridx].norm()
    return damp

@ti.kernel
def cal_total_force():
    for taridx in ti.grouped(f):
        f[taridx] = tina.V(0., -gravity, 0.) * mass[taridx] + cal_damp_force(taridx, idamping)
        if collision_ik > 0:
            f[taridx] += cal_collision_force(taridx)
        for d in ti.static(links):
            rest_length = L * float(d).norm()
            srcidx = min(max(taridx + d, 0), ti.Vector(NN) - 1)
            if all(srcidx == taridx):
                pass
            else:
                f[taridx] += cal_elastic_force(taridx, srcidx, stiffness, rest_length)

@ti.func
def cal_elastic_partial_on_x(taridx, srcidx, k, rest_length):
    """
    f = -k(||tarx-srcx||-restLength)*normalize(tarx-srcx)
    par(||x||)/par(x) = normalize(x)^T
    par(normalize(x))/par(x) = (I - normalize(x)*normalize(x)^T)/||x||
    """
    tarx = x[taridx]
    srcx = x[srcidx]
    cur_diff = tarx - srcx
    cur_length = cur_diff.norm() #ti.sqrt(cur_diff.dot(cur_diff))
    direction = cur_diff / cur_length
    deform_value = cur_length - rest_length
    cross = direction @ direction.transpose()
    elastic_partial_on_x = k * (deform_value / cur_length * (cross - Identity) - cross)
    return elastic_partial_on_x

@ti.func
def cal_damp_partial_on_v(taridx, kd):
    return -kd * Identity

@ti.func
def cal_collision_force_partial_on_v(i):
    force_par_v = Zero
    next_ball_pos = ball_pos[None] + ball_vel[None]*idt
    dp = x[i] - next_ball_pos
    dp2 = dp.norm_sqr()
    if dp2 <= ball_radius**2 + 1e-3:
        # a fun execise left for you: add angular velocity to the ball?
        # it should drag the cloth around with a specific friction rate.
        dv = v[i] - ball_vel[None]
        NoV = dv.dot(dp)
        #dv.dot(dp) on v = [dp[0], dp[1], dp[2]].transpose()
        if NoV < 0:
            p_NoV_on_v = dp.transpose()
            force_par_v = -(dp @ p_NoV_on_v) / dp2 * mass[i] / idt
    return force_par_v

@ti.kernel
def cal_total_force_partial():
    for taridx in ti.grouped(f):
        fParVMat[taridx] = cal_damp_partial_on_v(taridx, idamping)
        if collision_ik > 0:
            fParVMat[taridx] += cal_collision_force_partial_on_v(taridx)
        fParXMat[taridx] = Zero
        for d in ti.static(links):
            rest_length = L * float(d).norm()
            srcidx = min(max(taridx + d, 0), ti.Vector(NN) - 1)
            #print(taridx, srcidx)
            if all(srcidx == taridx):
                pass
            else:
                fParXMat[taridx] += \
                    cal_elastic_partial_on_x(taridx, srcidx, stiffness, rest_length)

@ti.kernel
def bulid_linear_system(): #Backward Euler
    for i in ti.grouped(x):
        if inv_mass[i] > 0.:
            AMat[i] = Identity - idt*inv_mass[i]*fParVMat[i] \
                - idt*idt*inv_mass[i]*fParXMat[i]
            BVec[i] = idt*inv_mass[i]*f[i] \
                + v[i] - idt*inv_mass[i]*fParVMat[i]@v[i]

@ti.kernel
def solve(): #Jacobi Solver
    for taridx in ti.grouped(x):
        if inv_mass[taridx] > 0.:
            for t in range(jacob_steps):
                for i in ti.static(range(3)):
                    r = BVec[taridx][i]
                    for j in ti.static(range(3)):
                        if i != j:
                            r -= AMat[taridx][i, j] * v[taridx][j]
                    v_next[taridx][i] = r / AMat[taridx][i, i]
                for i in ti.static(range(3)):
                    v[taridx][i] = v_next[taridx][i]

@ti.kernel
def integrate():
    for i in ti.grouped(x):
        if inv_mass[i] > 0.:
            if collision_ik == 0:
                dp = x[i] - ball_pos[None]
                dp2 = dp.norm_sqr()
                if dp2 <= ball_radius**2 + 1e-3:
                    # a fun execise left for you: add angular velocity to the ball?
                    # it should drag the cloth around with a specific friction rate.
                    dv = v[i] - ball_vel[None]
                    NoV = dv.dot(dp)
                    if NoV < 0:
                        v[i] -= NoV * dp / dp2
                apply_damping(i)
                x[i] += v[i]*idt
            else:
                if idamping == 0.:
                    apply_damping(i)
                x[i] += v[i]*idt
                dp = x[i] - ball_pos[None]
                dp2 = dp.norm_sqr()
                if dp2 <= ball_radius**2 + 1e-3:
                    x[i] = ball_pos[None] + dp * ti.rsqrt(dp2) * ball_radius
                    # dv = v[i] - ball_vel[None]
                    # NoV = dv.dot(dp)
                    # if NoV < 0:
                    #     x[i] += NoV * dp / dp2 * idt
        else:
            v[i] = x[i]*0.0

'''End Implicit'''

### Rendering GUI

# Hint: remove ibl=True if you find it compiles too slow...
scene = MyScene.MyRasterScene((1024, 768), smoothing=True, fxaa=True, ibl=False)
#tina.Scene((1024, 768), smoothing=True, texturing=True, ibl=True)
#scene = MyScene.MyPathTracingScene((1024, 768), smoothing=True, texturing=True, ibl=True)

mesh = tina.MeshNoCulling(tina.MeshGrid((N, N)))
ball = tina.MeshTransform(tina.PrimitiveMesh.sphere())

base_color_factor = 1./2.2#tina.Param(float, initial=1.5)
metallic0 = tina.Param(float, initial=0.0)
roughness0 = tina.Param(float, initial=0.4)
specular0 = tina.Param(float, initial=1.)

cloth_CookTolerance = {
    # 0:tina.PBR(basecolor=(ts.vec3(0.2, 0.8, 1.0) * 0.3)**base_color_factor, 
    #          metallic=metallic0, specular=specular0, roughness=roughness0),
    # 1:tina.PBR(basecolor=(ts.vec3(1.0, 0.95, 0.05) * 0.12 * 0.75 + ts.vec3(1.0, 0.95, 0.05) * 0.16 * 0.25)**base_color_factor,
    #          metallic=metallic0, specular=specular0, roughness=roughness0),
    2:tina.PBR(basecolor=(ts.vec3(1.0, 0.37, 0.3) * 0.035 * 0.9 + ts.vec3(1.0, 0.37, 0.3) * 0.2 * 0.1)**base_color_factor,
             metallic=metallic0, specular=specular0, roughness=roughness0),
    3:tina.PBR(basecolor=(ts.vec3(1.0, 0.37, 0.3) * 0.035 * 0.67 + ts.vec3(1.0, 0.37, 0.3) * 0.2 * 0.33)**base_color_factor,
             metallic=metallic0, specular=specular0, roughness=roughness0),
    4:tina.PBR(basecolor=(ts.vec3(0.1, 1.0, 0.4) * 0.2 * 0.86 + ts.vec3(1.0, 0.0, 0.1) * 0.6 * 0.14)**base_color_factor,
             metallic=metallic0, specular=specular0, roughness=roughness0),
    5:tina.PBR(basecolor=(ts.vec3(0.75, 0.02, 0.0) * 0.3 * 0.5 + ts.vec3(0.55, 0.02, 0.0) * 0.3 * 0.5)**base_color_factor,
             metallic=metallic0, specular=specular0, roughness=roughness0),
    }

cloth_Sadegi = {
    i:MyMaterial.SadegiClothMaterial(i) for i in cloth_CookTolerance.keys()
    }
    #tina.PBR(basecolor=tina.Texture('assets/cloth.jpg'))
    #tina.PBR(basecolor=tina.ChessboardTexture(size=0.2))
metal = tina.PBR(basecolor=[1.0, 0.9, 0.8], metallic=0.8, roughness=0.4)

scene.add_object(mesh, cloth_Sadegi[2])
scene.add_object(ball, metal)


if not isinstance(scene, tina.PTScene):
    for mat in cloth_CookTolerance.values():
        scene._ensure_material_shader(mat)
    for mat in cloth_Sadegi.values():
        scene._ensure_material_shader(mat)

is_implicit = False
simulating = True

def modify_patch_width(delta):
    MyMaterial.samplePatchWidth += delta
    MyMaterial.samplePatchWidth = max(1, MyMaterial.samplePatchWidth)
    print('samplePatchWidth =', MyMaterial.samplePatchWidth)

def flip_is_implicit(e, c):
    global is_implicit
    is_implicit = not is_implicit
    print('implicit' if is_implicit else 'explicit')

def flip_is_simulating(e, c):
    global simulating
    simulating = not simulating
    print('simulating' if simulating else 'not simulating')
    
def set_material(index, sadegi):
    if sadegi:
        if index in cloth_Sadegi:
            print('Sadegi', index)
            scene.change_material(mesh, cloth_Sadegi[index])
    else:
        if index in cloth_CookTolerance:
            print('Cook-Tolerance', index)
            scene.change_material(mesh, cloth_CookTolerance[index])

extra_input_events = \
{
    (ti.GUI.RELEASE, '0'):flip_is_implicit,
    (ti.GUI.RELEASE, ti.GUI.SPACE):flip_is_simulating,
}

extra_pressed_events = \
{
    ti.GUI.UP:lambda : modify_patch_width(1),
    ti.GUI.DOWN:lambda : modify_patch_width(-1),
}

class AuxiliaryFunc:
    def __init__(self, index):
        self.index = index
    
    def __call__(self, e, c):
        set_material(self.index, c(ti.GUI.CTRL))


for i in cloth_Sadegi:
    key = ascii(i + 1)
    extra_input_events[ti.GUI.RELEASE, key] = AuxiliaryFunc(i)

gui = ti.GUI('Mass Spring', scene.res, fast_gui=not isinstance(scene, tina.PTScene))
#gui = ti.GUI('Mass Spring', scene.res, fast_gui=False)
#gui = ti.GUI('Mass Spring', scene.res)

if not isinstance(scene, tina.PTScene):
    #base_color_factor.make_slider(gui, 'factor0', min=1., max=3.)
    roughness0.make_slider(gui, 'roughness0')
    metallic0.make_slider(gui, 'metallic0')
    specular0.make_slider(gui, 'specular0')

scene.init_control(gui,
        center=[0.0, 0.0, 0.0],
        theta=np.pi / 2 - np.radians(30),
        radius=1.5,
        input_events=extra_input_events,
        pressed_events=extra_pressed_events)

init()

@ti.kernel
def add():
    ball_pos[None] += ball_vel[None] * dt

@ti.kernel
def update_normal():
    mesh.pre_compute()

# for i in range(max(len(cloth_Sadegi), len(cloth_CookTolerance))):
#     set_material(i, False)
#     set_material(i, True)
# scene.render()
ball_vel_factor = 0.5
print('[Hint] Press ASWDEQ to move the ball, R to reset')
while gui.running:
    scene.input(gui)

    if simulating:
        if is_implicit:
            for i in range(isteps * factor):
                cal_total_force()
                cal_total_force_partial()
                bulid_linear_system()
                #print("before solve()")
                solve()
                #print("after solve()")
                integrate()
                
            for i in range(steps):
                add()
        else:
            for i in range(steps):
                substep_explicit()

    if gui.is_pressed('r'):
        init()

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
