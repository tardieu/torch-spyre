import sympy


class Tensor:
    def __init__(self, size, stride, device_size, stride_map):
        self.size = size
        self.stride = stride
        self.device_size = device_size
        self.stride_map = stride_map
        self.device_stride = [0] * len(self.device_size)  # to be computed
        self.relative_stride = [0] * len(self.device_size)  # to be computed
        self.dim_map = [-1] * len(self.device_size)  # to be computed

        # compute device_stride
        acc = 1
        for i in range(len(self.device_size) - 1, -1, -1):
            if self.stride_map[i] != 0:
                self.device_stride[i] = acc
                acc = acc * self.device_size[i]

        # compute relative_stride and dim_map
        acc = [0] * len(self.device_size)
        for i, hst in enumerate(self.stride):
            if self.size[i] == 1:
                continue
            for j, dst in enumerate(self.stride_map):
                if self.device_size[j] == 1:
                    continue
                if hst > acc[j] and hst <= dst:
                    acc[j] = hst
                    self.dim_map[j] = i
                    self.relative_stride[j] = dst // hst

    def h2d(self, h):
        """map host coordinates to device coordinates"""
        for i, c in enumerate(h):
            if c < 0 or c >= self.size[i]:
                raise IndexError
        d = [0] * len(self.dim_map)
        for j, i in enumerate(self.dim_map):
            if i != -1:
                d[j] = h[i] // self.relative_stride[j] % self.device_size[j]
        return d

    def d2h(self, d):
        """map device coordinates to host coordinates"""
        for i, c in enumerate(d):
            if c < 0 or c >= self.device_size[i]:
                raise IndexError
        h = [0] * len(self.size)
        for j, i in enumerate(self.dim_map):
            if i != -1:
                h[i] += d[j] * self.relative_stride[j]
                if h[i] > self.size[i]:
                    raise IndexError
        return h

    def compute_coordinates(self, var_ranges, index):
        """derive an array of coordinate expressions into a tensor from an index"""
        coordinates = [sympy.S.Zero] * len(self.size)
        vars = index.free_symbols
        for var in vars:
            if var_ranges[var] <= 1:
                continue
            term = index.subs({v: 0 for v in vars - {var}})
            step = term.subs(var, 1)
            limit = term.subs(var, var_ranges[var])
            primary_stride = 0
            primary_dim = -1
            for dim in range(len(self.size)):
                if self.size[dim] == 1:
                    continue
                st = self.stride[dim]
                if st > step and st < limit:
                    coordinates[dim] += var * step // st
                elif st <= step and st > primary_stride:
                    primary_stride = st
                    primary_dim = dim
            coordinates[primary_dim] += var * step // primary_stride
        return coordinates

    def compute_device_coordinates(self, var_ranges, index):
        """derive an array of coordinate expressions into a device tensor from an index"""
        host_coordinates = self.compute_coordinates(var_ranges, index)
        coordinates = [sympy.S.Zero] * len(self.device_size)
        for dim in range(len(self.device_size)):
            if self.dim_map[dim] == -1:
                continue
            expr = host_coordinates[self.dim_map[dim]]
            vars = expr.free_symbols
            for var in vars:
                term = expr.subs({v: 0 for v in vars - {var}})
                step = term.subs(var, 1)
                limit = term.subs(var, var_ranges[var])
                if (
                    limit > self.relative_stride[dim]
                    and step < self.relative_stride[dim] * self.device_size[dim]
                ):
                    coordinates[dim] += term // self.relative_stride[dim]
        return coordinates


print(vars(Tensor([512], [1], [512], [1])))
print(vars(Tensor([512], [1], [512], [1])))
print(vars(Tensor([4, 128], [128, 1], [4, 128], [128, 1])))
print(vars(Tensor([4, 128], [128, 1], [2, 4, 64], [64, 128, 1])))
print(vars(Tensor([3, 1, 256], [256, 256, 1], [1, 4, 3, 64], [256, 64, 256, 1])))
print(vars(Tensor([3, 256], [256, 1], [1, 4, 3, 64], [256, 64, 256, 1])))
print(vars(Tensor([1], [1], [1], [1])))
print(vars(Tensor([1, 1], [1, 1], [1, 1], [1, 1])))
print(vars(Tensor([1], [1], [1, 1], [1, 1])))
print(vars(Tensor([4, 64], [64, 1], [1, 4, 64], [64, 64, 1])))
print(vars(Tensor([4, 1], [1, 1], [1, 4, 64], [1, 1, -1])))
print(vars(Tensor([4], [1], [64], [1])))
print(vars(Tensor([4, 3], [3, 1], [1, 4, 64], [1, 3, 1])))

print(Tensor([4, 1], [1, 1], [1, 4, 64], [1, 1, -1]).d2h([0, 2, 0]))
print(Tensor([4, 3], [3, 1], [1, 4, 64], [1, 3, 1]).d2h([0, 2, 3]))
print(Tensor([4, 128], [128, 1], [2, 4, 64], [64, 128, 1]).d2h([1, 3, 7]))
