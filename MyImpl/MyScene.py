import taichi as ti
import numpy as np
import taichi_glsl as ts
#import os
#import sys

#sys.path.append(os.path.abspath('..'))

import tina

class MyControl(tina.Control):
    def __init__(self, gui, fov=60, is_ortho=False, blendish=False, 
                 input_events={},
                 pressed_events={}):
        super(MyControl, self).__init__(gui, fov, is_ortho, blendish)
        print('[Tina] Hint: LMB to orbit, RMB to pan, wheel to zoom')
        self.input_events = {
            (self.gui.PRESS, self.gui.TAB):\
                (lambda e, checker:\
                    self.press_tab(e, checker)),
            (self.gui.PRESS, self.gui.ESCAPE):\
                (lambda e, checker:\
                    self.press_escape(e, checker)),
            (self.gui.MOTION, self.gui.WHEEL):\
                (lambda e, checker:\
                    self.motion_wheel(e, checker)),
            # (self.gui.MOTION, self.gui.MOVE):\
            #     (lambda e, checker:\
            #         self.motion_move(e, checker)),
        }
        for key in input_events:
            self.input_events[key] = input_events[key]
        self.pressed_events = {}
        for key in pressed_events:
            self.pressed_events[key] = pressed_events[key]
        
    #override
    def on_event(self, e):

        def pressed_checker(key):
            return self.gui.is_pressed(key)

        ret = False

        if e.key in self.input_events:
            self.input_events[e.key](e, pressed_checker)
            ret = True
        elif (e.type, e.key) in self.input_events:
            self.input_events[e.type, e.key](e, pressed_checker)
            ret = True
        
        for key in self.pressed_events:
            if self.gui.is_pressed(key):
                self.pressed_events[key]()
                ret = True

        return ret

    def press_tab(self, e, pressed_checker):
        self.is_ortho = not self.is_ortho
        return True
        
    def press_escape(self, e, pressed_checker):
        self.gui.running = False

    def motion_wheel(self, e, pressed_checker):
        delta = e.delta[1] / 120
        self.on_wheel(delta, np.array(e.pos))

    def motion_move(self, e, pressed_checker):
        if self.gui.SHIFT in e.modifier and pressed_checker(ti.GUI.RMB):
            np_pos = np.array(e.pos)
            if self.last_mouse is not None:
                mouse_delta = np_pos - self.last_mouse
                delta = (mouse_delta[0], mouse_delta[1])
                self.on_pan(delta, np_pos)
            self.last_mouse = np_pos
        else:
            self.last_mouse = None

    

@ti.data_oriented
class MyRasterScene(tina.Scene):
    def __init__(self, res, **options):
        super(MyRasterScene, self).__init__(res, **options)
        if not self.ibl:
            @ti.materialize_callback
            def init_lights():
                self.lighting.set_lights([\
                    ([0, 1*ti.rsqrt(2), 1*ti.rsqrt(2), 0], [1, 1, 1]),
                    ([0, 1, 0, 0], [1, 1, 1])])
        
    #override
    def input(self, gui):
        '''
        :param gui: (GUI) GUI to recieve event from

        Feed inputs from the mouse drag events on GUI to control the camera
        '''

        if not hasattr(self, 'control'):
            self.control = tina.Control(gui)
        changed = self.control.apply_camera(self.engine)
        if changed:
            self.clear()
        return changed

    #override
    def init_control(self, gui, center=None, theta=None, phi=None, radius=None,
                     fov=60, is_ortho=False, blendish=False, 
                     input_events={},
                     pressed_events={}):
        '''
        :param gui: (GUI) the GUI to bind with
        :param center: (3 * [float]) the target (lookat) position
        :param theta: (float) the altitude of camera
        :param phi: (float) the longitude of camera
        :param radius: (float) the distance from camera to target
        :param is_ortho: (bool) whether to use orthogonal mode for camera
        :param blendish: (bool) whether to use blender key bindings
        :param fov: (bool) the initial field of view of the camera
        '''

        self.control = MyControl(gui, fov=fov, is_ortho=is_ortho, blendish=blendish,
                                 input_events=input_events,
                                 pressed_events=pressed_events)
        
        if center is not None:
            self.control.center[:] = center
        if theta is not None:
            self.control.theta = theta
        if phi is not None:
            self.control.phi = phi
        if radius is not None:
            self.control.radius = radius

    def change_material(self, object, material):
        self.objects[object]['material'] = material

            
@ti.data_oriented
class MyPathTracingScene(tina.PTScene):
    def __init__(self, res, **options):
        super(MyPathTracingScene, self).__init__(res, **options)
        #if not self.ibl:
        # @ti.materialize_callback
        # def init_lights():
        #     self.lighting.set_lights([\
        #         ([0, 1*ti.rsqrt(2), 1*ti.rsqrt(2), 0], [1, 1, 1]),
        #         ([0, 1, 0, 0], [1, 1, 1])])
        self.obj_mat_map = {}

    #override
    def add_object(self, object, material):
        super(MyPathTracingScene, self).add_object(object, material)
        self.obj_mat_map[object] = material
        
    #override
    def input(self, gui):
        return MyRasterScene.input(self, gui)

    #override
    def init_control(self, gui, center=None, theta=None, phi=None, radius=None,
                     fov=60, is_ortho=False, blendish=False, 
                     input_events={},
                     pressed_events={}):
        MyRasterScene.init_control(self, gui, center, theta, phi, radius,
                                   fov, is_ortho, blendish,
                                   input_events,
                                   pressed_events)
                                   
    def change_material(self, object, material):
        if material not in self.materials:
            self.materials.append(material)
            self.mtltab.add_material(material)
        prev_mtlid = self.materials.index(self.obj_mat_map[object])
        
        mtlid = self.materials.index(material)

        self.update_mtlid(prev_mtlid, mtlid)

    @ti.kernel
    def update_mtlid(self, src : ti.i32, tar : ti.i32):
        for i in self.geom.tracers[0].mtlids:
            if self.geom.tracers[0].mtlids[i] == src:
                self.geom.tracers[0].mtlids[i] = tar



if __name__ == '__main__':
    print(tina.__version__)