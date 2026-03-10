# This code demonstrate how to apply views to spyre tensor layouts.
#
# We assume a tensor with size, stride, device_size, and dim_map. In addition we
# assume an op var_ranges and indexer to index into the arg.
#
# For now, we assume indexer(0) = 0.
#
# We demonstrate:
# - how to map host coordinates to device coordinates and vice versa,
# - how to map an indexer to host coordinates,
# - how to map an indexer to device coordinates,
# - how to recompute the device_size and dim_map to avoid having multiple vars
#   in var_ranges indexing into the same device dimension.


class Padding(Exception):
    """Device coordinates do not map to host coordinated"""

    pass


class Tensor:
    def __init__(self, size, stride, device_size, dim_map):
        self.size = size
        self.stride = stride
        self.device_size = device_size
        self.dim_map = dim_map

        self.split = self.compute_split()

    def compute_split(self):
        """compute tensor split"""
        # split[i] is the stride of device dimension i w.r.t. host dimension
        # dim_map[i]
        s = [1] * len(self.size)
        split = [None] * len(self.dim_map)
        for i in range(len(self.dim_map) - 1, -1, -1):
            j = self.dim_map[i]
            if j is not None:
                split[i] = s[j]
                s[j] *= self.device_size[i]
        return split

    def h2d(self, h):
        """map host coordinates to device coordinates"""
        d = [0] * len(self.dim_map)
        for i in range(len(self.dim_map)):
            j = self.dim_map[i]
            if j is not None:
                d[i] = h[j] // self.split[i] % self.device_size[i]
        return d

    def d2h(self, d):
        """map device coordinates to host coordinates"""
        h = [0] * len(self.size)
        for i in range(len(self.dim_map)):
            j = self.dim_map[i]
            if j is not None:
                h[j] += d[i] * self.split[i]
                if h[j] > self.size[j]:
                    raise Padding
            elif d[i] > 0:
                raise Padding
        return h


class TensorArg:
    def __init__(self, tensor, var_ranges, indexer):
        self.tensor = tensor  # tensor
        self.var_ranges = var_ranges  # op range
        self.indexer = indexer  # indexer into tensor

        self.it_host_dim_map = self.compute_it_host_dim_map()
        self.it_device_dim_map = self.compute_it_device_dim_map()
        self.fixed_device_size, self.fixed_dim_map, self.fixed_it_device_dim_map = (
            self.fix_device_layout()
        )

    def compute_it_host_dim_map(self):
        """convert indexer into a map from var_ranges to host coordinates"""
        it_dim_map = [[()] for _ in range(len(self.var_ranges))]
        for i in range(len(self.var_ranges)):
            one = [0] * len(self.var_ranges)
            one[i] = 1
            step = self.indexer(one)
            if step == 0:
                # var does not occur in indexer
                continue
            limit = step * self.var_ranges[i]  # upper limit offset for this var
            max_stride_below = 0
            for j in range(len(self.tensor.size)):
                if self.tensor.size[j] == 1:
                    continue
                s = self.tensor.stride[j]
                if s > step and s < limit:
                    # var indexes into multiple host dimensions
                    # dim j has a stride>=1 w.r.t. var i
                    it_dim_map[i].append((j, 1, s // step))
                elif s <= step and s > max_stride_below:
                    # record stride as tentative max stride <= step
                    max_stride_below = s
                    # var i has a stride>=1 w.r.t. to dim j
                    it_dim_map[i][0] = (j, step // max_stride_below, 1)
        return it_dim_map

    def format_it_host_dim_map(self):
        exprs = ["0" for _ in range(len(self.tensor.size))]
        for i in range(len(self.it_host_dim_map)):
            for j, num, den in self.it_host_dim_map[i]:
                exprs[j] += f" + {num}*p{i}//{den}"  # % {self.tensor.size[j]}"
        return exprs

    def compute_it_device_dim_map(self):
        """convert indexer into a map from var_ranges to device coordinates"""
        it_dim_map = [[] for _ in range(len(self.var_ranges))]
        for i in range(len(self.var_ranges)):
            for j, num, den in self.it_host_dim_map[i]:
                for k in range(len(self.tensor.dim_map)):
                    if j != self.tensor.dim_map[k]:
                        # device dim k does not map to host dim j
                        continue
                    if (
                        self.var_ranges[i] * num // den > self.tensor.split[k]
                        and num // den
                        < self.tensor.split[k] * self.tensor.device_size[k]
                    ):
                        # var i indexes into device dim k
                        if num // den // self.tensor.split[k] > 0:
                            # var i has a stride>=1 w.r.t. device dim k
                            it_dim_map[i].append(
                                (k, num // den // self.tensor.split[k], 1)
                            )
                        else:
                            # device dim k has a stride>=1 w.r.t. var i
                            it_dim_map[i].append(
                                (k, 1, self.tensor.split[k] * den // num)
                            )
        return it_dim_map

    def format_it_device_dim_map(self):
        exprs = ["0" for _ in range(len(self.tensor.device_size))]
        for i in range(len(self.it_device_dim_map)):
            for j, num, den in self.it_device_dim_map[i]:
                exprs[j] += f" + {num}*p{i}//{den}"  # % {self.tensor.device_size[j]}"
        return exprs

    def fix_device_layout(self):
        # order indexing subexpressions for each coordinate in decreasing stride
        # order
        count = 0
        tmp = [[] for _ in range(len(self.tensor.device_size))]
        for i in range(len(self.tensor.device_size)):
            for k in range(len(self.it_device_dim_map)):
                for j, num, den in self.it_device_dim_map[k]:
                    if j != i:
                        continue
                    tmp[i].append((num, j, den, k))
                    count += 1
            tmp[i].sort()
            tmp[i].reverse()

        # split device dimensions with multiple indexing subexpressions into
        # consecutive device dimensions, fix dim_map and it_device_dim_map
        fixed_device_size = []
        fixed_dim_map = []
        fixed_it_device_dim_map = [None] * count
        x = 0
        for i in range(len(tmp)):
            current = 1
            for num, j, den, k in tmp[i]:
                fixed_device_size.append(self.tensor.device_size[i] // num // current)
                current *= self.tensor.device_size[i] // num
                fixed_dim_map.append(self.tensor.dim_map[i])
                fixed_it_device_dim_map[x] = (k, den)
                x += 1
        return fixed_device_size, fixed_dim_map, fixed_it_device_dim_map

    def format_fixed_it_device_dim_map(self):
        exprs = ["" for _ in range(len(self.fixed_device_size))]
        for i in range(len(self.fixed_it_device_dim_map)):
            j, den = self.fixed_it_device_dim_map[i]
            exprs[i] += f"p{j}//{den}"  # {self.fixed_device_size[i]}"
        return exprs


print("t1: B, S, E viewed as B, H, S, E/H")

t1 = Tensor([2, 256, 4096], [1048576, 4096, 1], [256, 64, 2, 64], [1, 2, 0, 2])

a1 = TensorArg(
    t1,
    [2, 32, 256, 128],
    lambda p: 1048576 * p[0] + 128 * p[1] + 4096 * p[2] + p[3],
)

print("""
t1 = Tensor([2, 256, 4096], [1048576, 4096, 1], [256, 64, 2, 64], [1, 2, 0, 2])

a1 = TensorArg(
    t1, [2, 32, 256, 128], lambda p: 1048576 * p[0] + 128 * p[1] + 4096 * p[2] +
    p[3],
)
""")

print("it_host_dim_map        ", a1.format_it_host_dim_map())
print("it_device_dim_map      ", a1.format_it_device_dim_map())
print("fixed_device_size      ", a1.fixed_device_size)
print("fixed_dim_map          ", a1.fixed_dim_map)
print("fixed_it_device_dim_map", a1.format_fixed_it_device_dim_map())

print()


print("t2: B, S, E viewed as B*S, E")

t2 = Tensor([2, 256, 4096], [1048576, 4096, 1], [256, 64, 2, 64], [1, 2, 0, 2])
a2 = TensorArg(t2, [512, 4096], lambda p: 4096 * p[0] + p[1])

print("""
t2 = Tensor([2, 256, 4096], [1048576, 4096, 1], [256, 64, 2, 64], [1, 2, 0, 2])
a2 = TensorArg(t2, [512, 4096], lambda p: 4096 * p[0] + p[1])
""")

print("it_host_dim_map        ", a2.format_it_host_dim_map())
print("it_device_dim_map      ", a2.format_it_device_dim_map())
print("fixed_device_size      ", a2.fixed_device_size)
print("fixed_dim_map          ", a2.fixed_dim_map)
print("fixed_it_device_dim_map", a2.format_fixed_it_device_dim_map())
