import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = position
        self.velocity = velocity

def compute_forces(bodies):
    '''
    Calculate force on each body given current positions

    :param bodies: all bodies in consideration
    :return: forces on the bodies, 1d-array of size n
    '''
    G = 1
    forces = []
    for i in range(len(bodies)):
        force = np.zeros(3)
        for j in range(len(bodies)):
            if i != j:
                r = bodies[j].position - bodies[i].position
                force += G * bodies[i].mass * bodies[j].mass * r / np.linalg.norm(r)**3
        forces.append(force)
    return np.array(forces)

def integrate(bodies, dt):
    '''
    Integrator that uses Verlet method

    :param bodies: all bodies in consideration
    :param dt: time step
    :return: None
    '''
    for i in range(len(bodies)):
        # x_1 = x_0 + dt * v_(1/2)
        bodies[i].position += dt * bodies[i].velocity
    forces = compute_forces(bodies)  # F(x_1)
    for i in range(len(bodies)):
        # v_(3/2) = v_(1/2) + F(x_1) * dt
        bodies[i].velocity += forces[i] * dt

def solve_n_body_problem(bodies, dt, steps):
    '''
    solving the problem using the integrator

    :param bodies: all bodies in consideration
    :param dt: time step
    :param steps: number of steps
    :return: None
    '''

    tot_m = 0    # total mass of bodies
    for body in bodies:
        tot_m += body.mass
    # initializations
    r_lst = []    # absolute positions
    r_cm_lst = []     # CM frame positions
    ini_forces = compute_forces(bodies)  # F(x_0)
    for i in range(len(bodies)):
        bodies[i].velocity += (ini_forces[i] / bodies[i].mass) * (dt / 2)  # v_(1/2)

    # integrate
    for _ in range(steps):
        current_r = []
        for body in bodies:
            current_r.append(body.position)
        r_lst.append(np.array(current_r))  # necessary to create new array object to avoid referencing same memory location
        # center of mass frame
        r_cm = 0     # position vector of center of mass
        for i in range(len(bodies)):
            r_cm += bodies[i].mass * current_r[i]
        r_cm /= tot_m
        current_r_cm = np.array(current_r) - r_cm         # positions in the CM frame
        r_cm_lst.append(current_r_cm)
        # proceed in time
        integrate(bodies, dt)
    r_arr = np.array(r_cm_lst)        # for observer's frame, replace by r_arr = np.array(r_lst)
    print(r_arr)  # overview of the positions
    flat_r_arr = r_arr.flatten()

    fig = plt.figure()
    # axis = plt.axes(xlim=(-5, 5), ylim=(-5, 5))
    # line, = axis.plot([], [], lw=3)
    # def init():
        # line.set_data([], [])
        # return line,

    for i in range(len(bodies)):
        """ def animate():
            mask = np.zeros(steps)
            for j in range(steps * len(bodies)):
                if (j % len(bodies) == i):
                    mask[j/len(bodies)] = 1

            x = r_arr[:,:,0][mask]
            y = r_arr[:,:,1][mask]
            line.set_data(x, y)
            return line,

        anim = FuncAnimation(fig, animate, init_func=init,
                             frames=200, interval=20, blit=True)
        """  # attempt to animate, will be fixed
        mask_x = np.zeros(len(flat_r_arr), dtype=int)
        mask_y = np.zeros(len(flat_r_arr), dtype=int)
        for j in range(len(flat_r_arr)):
            if j % (3 * len(bodies)) == 0:
                mask_x[j + 3 * i] = 1
            elif j % (3 * len(bodies)) == 1:
                mask_y[j + 3 * i] = 1
        print(mask_x)
        x = flat_r_arr[mask_x != 0]
        y = flat_r_arr[mask_y != 0]
        print(x, y)
        plt.scatter(x,y,s=1)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
    plt.show()

"""bodies = [Body(1.0, np.array([0.0,0.0,0.0]), np.array([1.0,0.0,0.0])),
          Body(2.0, np.array([1.0,1.0,0.0]), np.array([-1.0,0.0,0.0])),
          Body(1.0, np.array([0.0,-1.0,0.0]), np.array([0.0,1.0,0.0]))]
solve_n_body_problem(bodies, 0.01, 10000)

bodies = [Body(1.0, np.array([0.0,0.0,0.0]), np.array([1.0,0.0,0.0])),
          Body(2.0, np.array([1.0,1.0,0.0]), np.array([-1.0,0.0,0.0]))]
solve_n_body_problem(bodies, 0.01, 10000)

bodies = [Body(1.0, np.array([1.0,5.0,0.0]), np.array([-0.9,0.0,0.0])),
          Body(1.0, np.array([2.0,2.0,0.0]), np.array([0.0,1.5,0.0])),
          Body(1.0, np.array([0.0,0.0,-4.0]), np.array([0.0,0.0,3]))]
solve_n_body_problem(bodies, 0.01, 10000)

bodies = [Body(1, np.array([0.0,0.0,0.0]), np.array([1.0,0.0,0.0])),
          Body(5, np.array([1.0,1.0,0.0]), np.array([-1.0,0.0,0.0]))]
solve_n_body_problem(bodies, 0.01, 10000)"""

bodies = [Body(2, np.array([0.0,0.0,0.0]), np.array([-1.0,0.0,0.0])),
          Body(2, np.array([0.0,1.0,0.0]), np.array([1.0,0.0,0.0]))]
solve_n_body_problem(bodies, 0.001, 50000)

bodies = [Body(1, np.array([0.0,0.0,0.0]), np.array([1.0,0.0,0.0])),
          Body(2, np.array([1.0,1.0,0.0]), np.array([-1.0,0.0,0.0]))]
solve_n_body_problem(bodies, 0.0001, 150000)

bodies = [Body(1, np.array([0.0,1.0,0.0]), np.array([1.0,0.0,0.0])),
          Body(7, np.array([0.0,0.0,0.0]), np.array([0.0,0.0,0.0]))]
solve_n_body_problem(bodies, 0.0001, 50000)

'''bodies = [Body(1.0, np.array([0.0,0.0,0.0]), np.array([1.0,0.0,0.0])),
          Body(5.0, np.array([1.0,1.0,0.0]), np.array([-1.0,0.0,0.0])),
          Body(0.001, np.array([0.0,0.01,0.0]), np.array([0.0,0.0,0.0]))]
solve_n_body_problem(bodies, 0.0005, 100000)'''

'''bodies = [Body(1.0, np.array([0.0,0.0,0.0]), np.array([1.0,0.0,0.0])),
          Body(5.0, np.array([1.0,1.0,0.0]), np.array([-1.0,0.0,0.0])),
          Body(0.01, np.array([0.0,0.01,0.0]), np.array([0.0,0.0,0.0]))]
solve_n_body_problem(bodies, 0.001, 100000)'''
