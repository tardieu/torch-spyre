import sympy
from typing import Sequence


# A stride_map maps device dimensions to host strides
# stride_map[dim] = -1 for dimensions that do not exist in the host tensor
# stride_map[dim] = 0 is permitted to encode expanded dimensions


def compute_device_stride(
    device_size: Sequence[sympy.Expr], stride_map: Sequence[sympy.Expr]
):
    """Compute device strides"""
    device_stride = [0] * len(device_size)
    current_stride = 1
    for dim in range(len(device_size) - 1, -1, -1):
        if stride_map[dim] != 0:  # dimensions with stride 0 require no memory
            device_stride[dim] = current_stride
            current_stride *= device_size[dim]
    return device_stride


def compute_dim_map(
    size: Sequence[sympy.Expr],
    stride: Sequence[sympy.Expr],
    device_size: Sequence[sympy.Expr],
    stride_map: Sequence[sympy.Expr],
):
    """Compute dim_map"""
    max_stride_le = [0] * len(size)
    dim_map = [-1] * len(device_size)
    for i, hst in enumerate(stride):
        if size[i] == 1:
            continue  # dimensions of size 1 cannot reliably be mapped to hast dimensions
        for j, dst in enumerate(stride_map):
            if hst > max_stride_le[j] and hst <= dst:
                max_stride_le[j] = hst
                dim_map[j] = i
    return dim_map


def compute_padded_size(size: Sequence[sympy.Expr], stride: Sequence[sympy.Expr]):
    """Compute padded_size from size and stride"""
    padded_size = [0] * len(size)
    for dim in range(len(size)):
        min_stride_gt = max(stride) * size[dim]  # outermost dim is not padded
        for st in stride:
            if st > stride[dim] and st < min_stride_gt:
                min_stride_gt = st
        padded_size[dim] = min_stride_gt // stride[dim]
    return padded_size


def compute_coordinates(
    padded_size: Sequence[sympy.Expr],
    stride: Sequence[sympy.Expr],
    var_ranges: dict[sympy.Symbol, sympy.Expr],
    index: sympy.Expr,
) -> list[sympy.Expr]:
    """Compute an array of coordinate expressions from an index expression"""
    coordinates = [sympy.S.Zero] * len(stride)
    vars = index.free_symbols
    for var in vars:
        if var_ranges[var] <= 1:
            continue  # ranges of size 1 cannot reliably be mapped to coordinates
        term = index.subs({v: 0 for v in vars - {var}})
        step = term.subs(var, 1)
        limit = term.subs(var, var_ranges[var])
        primary_stride = 0
        primary_dim = -1
        for dim in range(len(stride)):
            if padded_size[dim] == 1:
                continue  # dimensions of size 1 cannot reliably be mapped to coordinates
            st = stride[dim]
            if st > step and st < limit:
                coordinates[dim] += var * step // st
            elif st <= step and st > primary_stride:
                primary_stride = st
                primary_dim = dim
        coordinates[primary_dim] += (
            var * step // primary_stride % padded_size[primary_dim]
        )
    return coordinates


if __name__ == "__main__":
    p0, p1, p2, p3 = sympy.symbols("p0 p1 p2 p3", integer=True)

    # B, S, E viewed as B, H, S, E/H on host
    print(
        compute_coordinates(
            [2, 256, 4096],
            [1048576, 4096, 1],
            {p0: 2, p1: 32, p2: 256, p3: 128},
            1048576 * p0 + 128 * p1 + 4096 * p2 + p3,
        )
    )

    # B, S, E viewed as B, H, S, E/H on device
    print(
        compute_coordinates(
            [256, 64, 2, 64],
            [4096, 64, 1048576, 1],
            {p0: 2, p1: 32, p2: 256, p3: 128},
            1048576 * p0 + 128 * p1 + 4096 * p2 + p3,
        )
    )

    # B, S, E viewed as B*S, E on host
    print(
        compute_coordinates(
            [2, 256, 4096],
            [1048576, 4096, 1],
            {p0: 512, p1: 4096},
            4096 * p0 + p1,
        )
    )

    # B, S, E viewed as B*S, E on device
    print(
        compute_coordinates(
            [256, 64, 2, 64],
            [4096, 64, 1048576, 1],
            {p0: 512, p1: 4096},
            4096 * p0 + p1,
        )
    )

    print(
        compute_coordinates(
            [3, 1, 128],
            [128, 128, 1],
            {p0: 3, p1: 128},
            128 * p0 + p1,
        )
    )
