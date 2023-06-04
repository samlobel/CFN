import numpy as np
import scipy.ndimage.filters
import scipy.ndimage
import scipy.stats


def get_truncated_normal(mean=0, sd=1.0, low=-1, upp=1):
    return scipy.stats.truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class NullSensor:
    def observe(self, s):
        return s

class MultiplySensor:
    def __init__(self, scale):
        self.scale = scale

    def observe(self, s):
        return s * self.scale

class AsTypeSensor:
    def __init__(self, dtype):
        self.dtype = dtype

    def observe(self, s):
        return s.astype(self.dtype)

class ClipSensor:
    def __init__(self, limit_min=None, limit_max=None, rescale=False):
        self.limit_min = limit_min
        self.limit_max = limit_max
        self.rescale = rescale

    def observe(self, s):
        return np.clip(s, self.limit_min, self.limit_max)

class RearrangeXYPositionsSensor:
    """Rearrange discrete x-y positions to break smoothness
    """
    def __init__(self, shape):
        self.shape = shape
        coords = np.asarray([[[x, y] for y in range(shape[1])] for x in range(shape[0])])
        self.encoding = np.random.permutation(coords.reshape(-1, 2)).reshape(coords.shape)

    def observe(self, s):
        try:
            obs = np.asarray([self.encoding[tuple(state)] for state in s])
        except TypeError:
            obs = self.encoding[tuple(s)]
        return obs

class OffsetSensor:
    def __init__(self, offset):
        self.offset = offset

    def observe(self, s):
        return s + self.offset

class NoisySensor:
    def __init__(self, sigma=0.1, truncation=None):
        self.sigma = sigma
        if truncation is not None:
            assert truncation > 0
        self.truncation = truncation

    def observe(self, s):
        if self.truncation is None:
            distribution = scipy.stats.norm(0, self.sigma)
        else:
            tr = self.truncation
            distribution = get_truncated_normal(0, self.sigma, low=-tr, upp=tr)

        if s.ndim == 0:
            n = distribution.rvs()
        elif s.ndim == 1:
            n = distribution.rvs(size=len(s))
        else:
            n = distribution.rvs(size=(s.shape[0], *s.shape[1:]))

        x = s + n
        return x

class ImageSensor:
    def __init__(self, range, pixel_density=1):
        assert isinstance(range, (tuple, list, np.ndarray))
        self.range = range
        self.size = (pixel_density * range[0][-1], pixel_density * range[1][-1])

    def observe(self, s):
        assert s.ndim > 0 and s.shape[-1] == 2
        if s.ndim == 1:
            s = np.expand_dims(s, axis=0)
        n_samples = s.shape[0]
        digitized = scipy.stats.binned_statistic_2d(
            s[:, 0],
            s[:, 1],
            np.arange(n_samples),
            statistic='count',
            bins=self.size,
            range=self.range,
            expand_binnumbers=True,
        )
        digitized = digitized[-1].transpose()
        x = np.zeros([n_samples, self.size[0], self.size[1]])
        for i in range(n_samples):
            x[i, digitized[i, 0] - 1, digitized[i, 1] - 1] = 1
        if n_samples == 1:
            x = x[0, :, :]
        return x

class MoveAxisSensor:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination

    def observe(self, s):
        return np.moveaxis(s, self.source, self.destination)

class ResampleSensor:
    def __init__(self, scale, order=0):
        assert type(scale) is int
        self.scale = scale
        self.order = order

    def observe(self, s):
        return np.kron(s, np.ones((self.scale, self.scale)))

class BlurSensor:
    def __init__(self, sigma=0.6, truncate=1.0):
        self.sigma = sigma
        self.truncate = truncate

    def observe(self, s):
        return scipy.ndimage.filters.gaussian_filter(s,
                                                     sigma=self.sigma,
                                                     truncate=self.truncate,
                                                     mode='nearest')

class PairEntangleSensor:
    def __init__(self, n_features, index_a=None, index_b=None, amount=1.0):
        # input:     X     Y     A     Z     S     B     T
        # output:    X     Y     A'    Z     S     B'    T
        # where [A',B']^T = R(theta) * [A, B]^T,
        #        R is a rotation-by-theta matrix,
        #        and theta = π/4 * amount
        assert n_features > 1, 'n_features must be > 1'
        assert 0 <= amount and amount <= 1, 'amount must be between 0 and 1'
        self.n_features = n_features
        self.rotation = np.pi / 4 * amount
        self.rot_matrix = np.asarray([[np.cos(self.rotation), -1 * np.sin(self.rotation)],
                                      [np.sin(self.rotation),
                                       np.cos(self.rotation)]])
        if index_b is not None:
            assert index_a is not None, 'Must specify index_a when specifying index_b'
            assert index_a != index_b, 'index_b cannot equal index_a (value {})'.format(index_a)
            self.index_b = index_b
        if index_a is not None:
            self.index_a = index_a
        else:
            self.index_a = np.random.randint(n_features)
        if index_b is None:
            self.index_b = np.random.choice([i for i in range(n_features) if i != self.index_a])

    def observe(self, s):
        s_flat = np.copy(s).reshape(-1, self.n_features)
        a = s_flat[:, self.index_a]
        b = s_flat[:, self.index_b]
        x = np.stack((a, b), axis=0)
        x_rot = np.matmul(self.rot_matrix, x)
        a, b = map(lambda a: np.squeeze(a, axis=0), np.split(x_rot, 2, axis=0))
        s_flat[:, self.index_a] = a
        s_flat[:, self.index_b] = b
        return s_flat.reshape(s.shape)

class PermuteAndAverageSensor:
    def __init__(self, n_features, n_permutations=1):
        self.n_features = n_features
        self.permutations = [np.arange(n_features)] + [
            np.random.permutation(n_features) for _ in range(n_permutations)
        ]

    def observe(self, s):
        s_flat = s.reshape(-1, self.n_features)
        output = np.zeros_like(s_flat)
        for p in self.permutations:
            sp_flat = np.take(s_flat, p, axis=1)
            sp = sp_flat.reshape(s.shape)
            output += sp
        return output / len(self.permutations)

# class TorchSensor:

#     def __init__(self, dtype=torch.float32):

#         self.dtype = dtype

#     def observe(self, s):
#         return torch.as_tensor(s, dtype=self.dtype)

class UnsqueezeSensor:
    def __init__(self, dim=-1):
        self.dim = dim

    def observe(self, s):
        return s.unsqueeze(dim=self.dim)

class SensorChain:
    def __init__(self, sensors):
        self.sensors = sensors

    def observe(self, s):
        for sensor in self.sensors:
            s = sensor.observe(s)
        return s
