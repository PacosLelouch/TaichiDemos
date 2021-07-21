import taichi as ti
import taichi_glsl as ts
import numpy as np
import tina


@ti.data_oriented
class MyXPBDSolver:
    def __init__(self, mesh, ball):
        self.mesh = mesh
        self.ball = ball
        '''Start numerical settings'''
        N = mesh.res[0]
        self.N = N
        self.W = 1
        self.rest_length = self.W / self.N
        self.mass0 = 1.#0.5#1. / (N*N)#2.#0.1#0.1
        self.gravity = 0.1#0.1#10.#0.2#1.#0.5
        self.stiffness = 1000.#1000.#1000.#3600.#3200.#1600.#2400.# * N / W
        self.inv_stiffness = 0. if self.stiffness == 0. else 1. / self.stiffness
        self.ball_radius = 0.4
        self.damping = 1.#2#2#200
        
        self.isteps = 2#2#3#3#15
        steps = 15#30

        self.opt_steps = 3#5#3#5#10#60#30
        self.dt = 5e-4 * steps#1e-3 * steps#5e-4

        links = [(-1, 0), (1, 0), (0, -1), (0, 1), \
                 (-1, -1), (1, -1), (-1, 1), (1, 1), \
                 (-2, 0), (2, 0), (0, -2), (0, 2), ]
        self.links = [ti.Vector(_) for _ in links]

        self.n_links = len(links)
        self.M = N * N
        self.S = self.M * self.n_links

        self.inv_collision_stiffness = 0.#-1e-9

        '''End numerical settings'''
        '''Start tensor settings'''

        # self.Z3 = ti.Matrix([[0.,0.,0.],
        #                      [0.,0.,0.],
        #                      [0.,0.,0.]])
        # self.I3 = ti.Matrix([[1.,0.,0.],
        #                      [0.,1.,0.],
        #                      [0.,0.,1.]])

        self.ball_pos = ti.Vector.field(3, float, ())
        self.ball_vel = ti.Vector.field(3, float, ())

        self.collision_lambda = ti.field(float, ())

        self.mass = ti.field(float, self.M)
        self.inv_mass = ti.field(float, self.M)

        self.x_ti = ti.Vector.field(3, float, self.M)
        self.v_ti = ti.Vector.field(3, float, self.M)
        self.fext_ti = ti.Vector.field(3, float, self.M)

        self.p_ti = ti.Vector.field(3, float, self.M)
        #self.p0_ti = ti.Vector.field(3, float, self.M)# Jacobi/Chebyshev instead of Gauss-Seidel?
        
        self.rest_ti = ti.field(float)
        self.lambda_ti = ti.field(float)

        # connect_block = ti.root.pointer(ti.ij, (self.M // 4, self.M // 4))
        # connect_block.pointer(ti.ij, (4, 4)).place(self.rest_ti, self.lambda_ti)

        ti.root.dense(ti.i, self.S).place(self.rest_ti, self.lambda_ti)
        '''End tensor settings'''

    @property
    def DT(self):
        return self.dt

    def init_data(self):
        self.init_ti()

    @ti.kernel
    def init_ti(self):
        self.ball_pos[None] = ti.Vector([0.0, +0.0, 0.0])
        for i in self.x_ti:
            it = self.get_ij(i)
            fix = (it[0] == 0 and it[1] == 0 or it[0] == self.N - 1 and it[1] == 0)
            #fix = (it[1] == 0)
            self.mass[i] = ti.select(fix, 0., self.mass0)
            self.inv_mass[i] = ti.select(fix, 0., 1. / self.mass[i])
            m, n = (it + 0.5) * self.rest_length - 0.5
            self.x_ti[i] = ti.Vector([m, 0.6, n])
            self.v_ti[i] = ti.Vector([0.0, 0.0, 0.0])
            self.fext_ti[i] = ti.Vector([0., -self.mass[i]*self.gravity, 0.])
            
        for i, j in ti.ndrange(self.N, self.N):
            I_vec = ti.Vector([i, j])
            I = self.get_index(i, j)
            index = 0
            for D in ti.static(self.links):
                J0_vec = I_vec + D
                J_vec = min(max(J0_vec, 0), ti.Vector([self.N, self.N]) - 1)
                J = self.get_index(J_vec.x, J_vec.y)
                spidx = self.get_spring(I, index)
                index += 1
                if all(J0_vec == J_vec):
                    length = self.rest_length * (J_vec - I_vec).norm()

                    #self.rest_ti[I, J] = length
                    #self.lambda_ti[I, J] = 0.
                    self.rest_ti[spidx] = length
                    self.lambda_ti[spidx] = 0.

    @ti.kernel
    def test_sparse_ti(self, count : ti.template()):
        count[None] = 0
        for I in ti.grouped(self.rest_ti):
            count[None] += 1
        print(count)

    '''Sparse structure is still not suitable, use dense instead.'''
    def test_sparse(self, count):
        self.test_sparse_ti(count)
        rest_np = self.rest_ti.to_numpy()
        print(rest_np, rest_np.shape)
            
    def reinit_data(self):
        self.reinit_ti()
    
    @ti.kernel
    def reinit_ti(self):
        self.ball_pos[None] = ti.Vector([0.0, +0.0, 0.0])
        for i in self.x_ti:
            it = self.get_ij(i)
            m, n = (it + 0.5) * self.rest_length - 0.5
            self.x_ti[i] = ti.Vector([m, 0.6, n])
            self.v_ti[i] = ti.Vector([0.0, 0.0, 0.0])

    def step(self):
        for __ in range(self.isteps):
            # The step of XPBD
            self.predict_position()
            for _ in range(self.opt_steps):
                self.project_constraint()
            self.update_current_state()
            self.copy_x()
            self.ball.set_transform(tina.translate(self.ball_pos[None].value) @ tina.scale(self.ball_radius))
        
    @ti.func
    def move_ball(self, dt):
        self.collision_lambda[None] = 0.
        self.ball_pos[None] += self.ball_vel[None] * dt

    @ti.kernel
    def predict_position(self):
        self.move_ball(self.DT)
        for i in self.x_ti:
            self.p_ti[i] = self.x_ti[i] \
                         + self.v_ti[i] * self.DT \
                         + self.fext_ti[i] * (self.inv_mass[i]*self.DT*self.DT)
        for I in ti.grouped(self.lambda_ti):
            self.lambda_ti[I] = 0.

    @ti.kernel
    def project_constraint(self):
        self.project_spring_constraint(self.inv_stiffness, self.damping)
        self.project_collision_constraint(self.inv_collision_stiffness, self.damping)

    @ti.kernel
    def update_current_state(self):
        for i in self.x_ti:
            self.v_ti[i] = (self.p_ti[i] - self.x_ti[i]) / self.DT
            self.x_ti[i] = self.p_ti[i]

    @ti.func
    def project_spring_constraint(self, inv_stiffness, damping=0.):
        # for i in self.p_ti:
        #     self.p0_ti[i] = self.p_ti[i]

        # for i in self.x_ti:
        #     for D in ti.static(self.links):
        #         pass
        #for i, j in ti.grouped(self.rest_ti): # For each spring constraint

        for i0, j0 in ti.ndrange(self.N, self.N):
            #i0 = self.N - 1 - i0
            #j0 = self.N - 1 - j0
            I_vec = ti.Vector([i0, j0])
            i = self.get_index(i0, j0)
            
            inv_mass = self.inv_mass[i]

            if inv_mass > 0.:
                index = 0
                delta_p = ti.Vector([0., 0., 0.])
                for D in ti.static(self.links):
                    J0_vec = I_vec + D
                    J_vec = min(max(J0_vec, 0), ti.Vector([self.N, self.N]) - 1)
                    j = self.get_index(J_vec.x, J_vec.y)
                    spidx = self.get_spring(i, index)
                    index += 1
                    if all(J0_vec == J_vec):
                        #disp = self.p0_ti[i] - self.p0_ti[j]
                        disp = self.p_ti[i] - self.p_ti[j]
                        length = disp.norm()
                        #cons = length - self.rest_ti[i, j]
                        cons = length - self.rest_ti[spidx]
                        cons_dp_tr = disp / length
                        
                        pdiff = self.p_ti[i] - self.x_ti[i]
                        damp_value = cons_dp_tr.dot(pdiff)
                        
                        alpha = inv_stiffness / (self.DT*self.DT)
                        beta = damping * (self.DT*self.DT)
                        gamma = (alpha*beta)/self.DT

                        #delta_lambda = (-cons - alpha*self.lambda_ti[i, j] - gamma*damp_value) \
                        #            / ((1.+gamma)*cons_dp_tr.dot(cons_dp_tr)*inv_mass + alpha)
                        delta_lambda = (-cons - alpha*self.lambda_ti[spidx] - gamma*damp_value) \
                                     / ((1.+gamma)*cons_dp_tr.dot(cons_dp_tr)*inv_mass + alpha)
                        delta_p += cons_dp_tr * (inv_mass*delta_lambda)

                        #self.lambda_ti[i, j] += delta_lambda
                        self.lambda_ti[spidx] += delta_lambda
                self.p_ti[i] += delta_p
            
    @ti.func
    def project_collision_constraint(self, inv_stiffness, damping=0.):

        for i in self.p_ti:
            if self.inv_mass[i] > 0.:
                disp = self.p_ti[i] - self.ball_pos[None]
                length = disp.norm()
                cons = min(length - (self.ball_radius + 1e-2), 0.)
                cons_dp_tr = disp / length

                pdiff = self.p_ti[i] - self.x_ti[i]
                damp_value = cons_dp_tr.dot(pdiff)

                alpha = inv_stiffness / (self.DT*self.DT)
                beta = damping * (self.DT*self.DT)
                gamma = (alpha*beta)/self.DT

                # Inequation, alpha = 0.?
                
                # delta_lambda = -cons / self.inv_mass[i]
                # delta_x = cons_dp_tr * (self.inv_mass[i]*delta_lambda/cons_dp_tr.dot(cons_dp_tr))
                delta_lambda = (-cons - alpha*self.collision_lambda[None] - gamma*damp_value) \
                             / ((1.+gamma)*cons_dp_tr.dot(cons_dp_tr)*self.inv_mass[i] + alpha)
                delta_x = cons_dp_tr * (self.inv_mass[i]*delta_lambda)
                
                # delta_x = cons_dp_tr * -cons / cons_dp_tr.dot(cons_dp_tr)
                
                self.collision_lambda[None] += delta_lambda
                self.p_ti[i] += delta_x
    
    @ti.kernel
    def copy_x(self):
        for i, j in ti.grouped(self.mesh.pos):
            index = self.get_index(i, j)
            self.mesh.pos[i, j] = self.x_ti[index]
        self.mesh.pre_compute()

    @ti.func
    def get_index(self, i, j):
        return i * self.N + j
        
    @ti.func
    def get_ij(self, index):
        i = index // self.N
        j = index - i * self.N
        return tina.V(i, j)

    @ti.func
    def get_spring(self, i, j):
        return i * self.n_links + j
    