import sympy


class Tensor:
    def __init__(self, size, stride, device_size, stride_map):
        self.size = size
        self.stride = stride
        self.device_size = device_size
        self.stride_map = stride_map
        self.device_stride = [0] * len(self.device_size)  # to be computed
        self.unpadded_size = [0] * len(self.size)  # to be computed

        # compute device_stride
        acc = 1
        for dim in range(len(self.device_size) - 1, -1, -1):
            if self.stride_map[dim] != 0:
                self.device_stride[dim] = acc
                acc = acc * self.device_size[dim]

        # compute unpadded_size
        for dim in range(len(self.size)):
            top = max(self.stride) * self.size[dim]
            for st in stride:
                if st > stride[dim] and st < top:
                    top = st
            self.unpadded_size[dim] = top // self.stride[dim]

    def h2d(self, h):
        """map host coordinates to device coordinates"""
        offset = 0
        for dim, c in enumerate(h):
            if c < 0 or c >= self.size[dim]:
                raise IndexError
            offset += c * self.stride[dim]
        d = [0] * len(self.device_size)
        for dim in range(len(self.device_size)):
            if self.stride_map[dim] > 0:
                d[dim] = offset // self.stride_map[dim] % self.device_size[dim]
        return d

    def d2h(self, d):
        """map device coordinates to host coordinates"""
        offset = 0
        for dim, c in enumerate(d):
            if c < 0 or c >= self.device_size[dim]:
                raise IndexError
            offset += c * max(0, self.stride_map[dim])
        h = [0] * len(self.size)
        for dim in range(len(self.size)):
            h[dim] += offset // self.stride[dim] % self.unpadded_size[dim]
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
            coordinates[primary_dim] += (
                var * step // primary_stride  # % self.unpadded_size[primary_dim]
            )
        return coordinates

    def compute_device_coordinates(self, var_ranges, index):
        """derive an array of coordinate expressions into a device tensor from an index"""
        coordinates = [sympy.S.Zero] * len(self.device_size)
        vars = index.free_symbols
        for var in vars:
            if var_ranges[var] <= 1:
                continue
            term = index.subs({v: 0 for v in vars - {var}})
            step = term.subs(var, 1)
            limit = term.subs(var, var_ranges[var])
            primary_stride = 0
            primary_dim = -1
            for dim in range(len(self.device_size)):
                if self.device_size[dim] == 1:
                    continue
                st = self.stride_map[dim]
                if st > step and st < limit:
                    coordinates[dim] += var * step // st
                elif st <= step and st > primary_stride:
                    primary_stride = st
                    primary_dim = dim
            coordinates[primary_dim] += (
                var * step // primary_stride  # % self.device_size[primary_dim]
            )
        return coordinates


print(vars(Tensor([512, 256], [384, 1], [196608], [1])))

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
print(vars(Tensor([4, 128], [128, 1], [2, 4, 64], [64, 128, 1])))

print(Tensor([4, 1], [1, 1], [1, 4, 64], [1, 1, -1]).d2h([0, 2, 0]))
print(Tensor([4, 3], [3, 1], [1, 4, 64], [1, 3, 1]).d2h([0, 2, 3]))
print(Tensor([4, 128], [128, 1], [2, 4, 64], [64, 128, 1]).d2h([1, 3, 7]))

print(Tensor([512, 256], [384, 1], [196608], [1]).d2h([255]))

p0, p1, p2, p3 = sympy.symbols("p0 p1 p2 p3", integer=True)

print(
    Tensor(
        [2, 256, 4096], [1048576, 4096, 1], [256, 64, 2, 64], [4096, 64, 1048576, 1]
    ).compute_coordinates(
        {p0: 2, p1: 32, p2: 256, p3: 128},
        1048576 * p0 + 128 * p1 + 4096 * p2 + p3,
    )
)

print(
    Tensor(
        [2, 256, 4096], [1048576, 4096, 1], [256, 64, 2, 64], [4096, 64, 1048576, 1]
    ).compute_device_coordinates(
        {p0: 2, p1: 32, p2: 256, p3: 128},
        1048576 * p0 + 128 * p1 + 4096 * p2 + p3,
    )
)


print(
    Tensor(
        [2, 256, 4096], [1048576, 4096, 1], [256, 64, 2, 64], [4096, 64, 1048576, 1]
    ).compute_coordinates(
        {p0: 512, p1: 4096},
        4096 * p0 + p1,
    )
)

print(
    Tensor(
        [2, 256, 4096], [1048576, 4096, 1], [256, 64, 2, 64], [4096, 64, 1048576, 1]
    ).compute_device_coordinates(
        {p0: 512, p1: 4096},
        4096 * p0 + p1,
    )
)

print(
    Tensor(
        [256, 4096], [4096, 1], [64, 256, 64], [64, 4096, 1]
    ).compute_device_coordinates(
        {p0: 256, p1: 4096, p2: 1024},
        4096 * p0 + p1,
    )
)

print(
    Tensor(
        [1024, 4096], [4096, 1], [64, 1024, 64], [64, 4096, 1]
    ).compute_device_coordinates(
        {p0: 256, p1: 4096, p2: 1024},
        p1 + 4096 * p2,
    )
)

print(
    Tensor([4, 1], [1, 1], [1, 4, 64], [-1, 1, -1]).compute_device_coordinates(
        {p0: 4},
        p0,
    )
)
