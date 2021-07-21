import taichi as ti
import taichi_glsl as ts
import numpy as np
import tina
import MyImpl.MyScene as MyScene
import MyImpl.MyMaterial as MyMaterial
import MyImpl.MyXPBDSolver as MyXPBDSolver

ti.init(arch=ti.gpu, advanced_optimization=True, device_memory_GB=2.0)
#ti.init(arch=ti.cpu, advanced_optimization=True, device_memory_GB=2.0)

### Parameters

N = 128#64#128#16#128

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

simulating = True

def modify_patch_width(delta):
    MyMaterial.samplePatchWidth += delta
    MyMaterial.samplePatchWidth = max(1, MyMaterial.samplePatchWidth)
    print('samplePatchWidth =', MyMaterial.samplePatchWidth)

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


ball_vel_factor = 0.5

count = ti.field(ti.i32, ())

solver = MyXPBDSolver.MyXPBDSolver(mesh, ball)
solver.init_data()
#solver.test_sparse(count)

print('[Hint] Press ASWDEQ to move the ball, R to reset')
while gui.running:
    scene.input(gui)

    if simulating:
        solver.step()

    if gui.is_pressed('r'):
        solver.reinit_data()

    if gui.is_pressed('w'):
        solver.ball_vel[None] = tina.V(0, 0, -1) * ball_vel_factor
    elif gui.is_pressed('s'):
        solver.ball_vel[None] = tina.V(0, 0, +1) * ball_vel_factor
    elif gui.is_pressed('e'):
        solver.ball_vel[None] = tina.V(0, +1, 0) * ball_vel_factor
    elif gui.is_pressed('q'):
        solver.ball_vel[None] = tina.V(0, -1, 0) * ball_vel_factor
    elif gui.is_pressed('a'):
        solver.ball_vel[None] = tina.V(-1, 0, 0) * ball_vel_factor
    elif gui.is_pressed('d'):
        solver.ball_vel[None] = tina.V(+1, 0, 0) * ball_vel_factor
    else:
        solver.ball_vel[None] = tina.V(0, 0, 0)

    # if isinstance(scene, tina.PTScene):
    #     scene.render(nsteps=16)
    # else:
    #     scene.render()
    scene.render()
    gui.set_image(scene.img)
    gui.show()
