import os
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing as mp


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--save_path', type=str, required=True, help="Path to the config YAML file")
    args = parser.parse_args()
    
    return args


class ChargedParticlesSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=1., vel_norm=0.5,
                 interaction_strength=1., noise_var=0.):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._charge_types = np.array([-1., 0., 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _energy(self, loc, vel, edges):

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] / dist
            return U + K

    def _clamp(self, loc, vel):
        '''
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        '''
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def sample_trajectory(self, T=10000, sample_freq=10,
                          charge_prob=[1. / 2, 0, 1. / 2]):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Sample edges
        charges = np.random.choice(self._charge_types, size=(self.n_balls, 1),
                                   p=charge_prob)
        edges = charges.dot(charges.transpose())
        # Initialize location and velocity
        loc = np.zeros((T_save, 3, n))
        vel = np.zeros((T_save, 3, n))
        loc_next = np.random.randn(3, n) * self.loc_std
        vel_next = np.random.randn(3, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        
        if self.box_size is not None:
            loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            # half step leapfrog
            l2_dist_power3 = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)

            # size of forces up to a 1/|r| factor
            # since I later multiply by an unnormalized r vector
            forces_size = self.interaction_strength * edges / l2_dist_power3
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[2, :],
                                       loc_next[2, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                if self.box_size is not None:
                    loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                l2_dist_power3 = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()),
                    3. / 2.)
                forces_size = self.interaction_strength * edges / l2_dist_power3
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[2, :],
                                           loc_next[2, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 3, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 3, self.n_balls) * self.noise_var
            return loc, vel, edges, charges
        

def run_simulation(args):
    simulator, T, sample_freq = args
    loc, vel, edges, charges = simulator.sample_trajectory(T=T, sample_freq=sample_freq)
    return loc, vel, edges, charges

def parallel_simulations(simulator, num_simulations, T=5000, sample_freq=100):
    loc_complete = []
    vel_complete = []
    edges_complete = []
    charges_complete = []

    args_list = [(simulator, T, sample_freq) for _ in range(num_simulations)]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(run_simulation, args_list), total=num_simulations))

    # Collect results
    for loc, vel, edges, charges in results:
        loc_complete.append(loc)
        vel_complete.append(vel)
        edges_complete.append(edges)
        charges_complete.append(charges)

    return loc_complete, vel_complete, edges_complete, charges_complete


if __name__ == '__main__':
    args = parse_args()
    save_path = args.save_path

    simulator = ChargedParticlesSim(n_balls=5, box_size=None)

    # save_path = 'nbody_3000'
    os.makedirs(save_path)

    # TRAIN
    loc_complete, vel_complete, edges_complete, _ = parallel_simulations(simulator, 500, 5000, 100) # 3000

    loc_complete = np.stack(loc_complete, axis=0)
    vel_complete = np.stack(vel_complete, axis=0)
    edges_complete = np.stack(edges_complete, axis=0)

    print(loc_complete.shape, vel_complete.shape, edges_complete.shape)

    np.save(f'{save_path}/loc_train.npy', loc_complete)
    np.save(f'{save_path}/vel_train.npy', vel_complete)
    np.save(f'{save_path}/edges_train.npy', edges_complete)

    # VAL
    loc_complete, vel_complete, edges_complete, _ = parallel_simulations(simulator, 300, 5000, 100) # 2000

    loc_complete = np.stack(loc_complete, axis=0)
    vel_complete = np.stack(vel_complete, axis=0)
    edges_complete = np.stack(edges_complete, axis=0)

    print(loc_complete.shape, vel_complete.shape, edges_complete.shape)

    np.save(f'{save_path}/loc_val.npy', loc_complete)
    np.save(f'{save_path}/vel_val.npy', vel_complete)
    np.save(f'{save_path}/edges_val.npy', edges_complete)

    # TEST
    loc_complete, vel_complete, edges_complete, _ = parallel_simulations(simulator, 300, 5000, 100) # 2000

    loc_complete = np.stack(loc_complete, axis=0)
    vel_complete = np.stack(vel_complete, axis=0)
    edges_complete = np.stack(edges_complete, axis=0)

    print(loc_complete.shape, vel_complete.shape, edges_complete.shape)

    np.save(f'{save_path}/loc_test.npy', loc_complete)
    np.save(f'{save_path}/vel_test.npy', vel_complete)
    np.save(f'{save_path}/edges_test.npy', edges_complete)