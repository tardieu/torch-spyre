# Stickification

## Tensors

A tensor has:
- a host `size` array and `stride` array with equal lengths,
- a `device_size` array and `dim_map` array with equal lengths.

The `rank` of a tensor is `len(size)`.

The number of elements in the tensor is `prod(size)`.

The `device_size` length is always greater than or equal to the `rank`.

## Strides

Per-dimension host strides are not required to be decreasing or increasing.

Per-dimension device strides are implicit, decreasing, and derived from the
`device_size`. The `device_stride[i]` of device dimension `i` is
`prod(device_size[i+1:])`.

## Offsets

An offset (on host or device) is a non-negative displacement respective to a
base address. An increment of one correspond to a displacement of one element
(not one byte).

The host offset of the element with coordinate vector `h` such that `0<=h<size`
is `dot(h, stride)`. Two distinct coordinate vectors must not map to the same
offset.

Host offsets are in range `[0, max(stride)*size[argmax(stride)])`.

The device offset of the element with coordinate vector `d` such that
`0<=d<device_size` is `dot(d, device_stride)`. Two distinct coordinate vectors
must not map to the same offset.

Device offsets are in range `[0, prod(device_size))`.

## Coordinate Mapping

The `dim_map` maps device dimensions to host dimensions. Each host dimensions
must occur at least once in `dim_map`. A `dim_map` element may be `None` to
denote a synthetic dimension that does not map to a host dimension.

For each host dimension `j` the product of all `device_size[i]` elements such
that `dim_map[i]==j` must be greater than or equal to `size[j]`.

To map host coordinates to device coordinates and vice versa we first compute
splits:
- `current_split[i]` is the host stride for device dimension `i` in host
  dimension `dim_map[i]`.
- `next_split[i]` is `current_split[i]*device_size[i]`.

```python
def splits(t):
    s = [1] * len(t.size)
    current_split = [None] * len(t.dim_map)
    next_split = [None] * len(t.dim_map)
    for i in range(len(t.dim_map)-1, -1, -1):
        j = t.dim_map[i]
        if j is not None:
            current_split[i] = s[j]
            s[j] *= t.device_size[i]
            next_split[i] = s[j]
    return current_split, next_split
```

The host element with coordinate vector `h` such that `0<h<size` corresponds to
device element with coordinate vector `d` obtained as follows:
```python
def h2d(t, h):
    current_split, next_split = splits(t)
    d = [0] * len(t.dim_map)
    for i in range(len(t.dim_map)):
        j = t.dim_map[i]
        if j is not None:
            d[i] = h[j] % next_split[i] // current_split[i]
    return d
```

The device element with coordinate vector `d` such that `0<d<device_size`
corresponds either to padding or a host element with coordinate vector `h`
obtained as follows:
```python
def d2h(t, d):
    current_split, next_split = splits(t)
    h = [0] * len(t.size)
    for i in range(len(t.dim_map)):
        j = t.dim_map[i]
        if j is not None:
            h[j] += d[i] * current_split[i]
            if h[j] > t.size[j]:
                raise Padding
        elif d[i] > 0:
            raise Padding
    return h
```

## Ops

An operation has:
- a `var_ranges` array,
- an `indexer` to access each argument tensor.

An indexer is a function that takes a coordinate vector `v` of length
`len(var_ranges)` such that `0<=v<var_ranges` and returns a host offset.

The indexer function is assumed be affine with the coefficient of each
coordinate being a multiple of the largest per-dimension tensor stride less than
or equal to the coefficient.

From each argument tensor, we compute an `it_dim_map` mapping the operation
dimensions to host dimensions:
```python
def analyze(t, var_ranges, indexer):
    zero_index = indexer([0] * len(var_ranges))
    it_dim_map = [None] * len(var_ranges)
    relative_stride = [None] * len(var_ranges)
    for i in range(var_ranges):
        one = [0] * len(var_ranges)
        one[i] = 1
        delta = indexer(one) - zero_index
        stride_match = None
        for j in range(len(t.size)):
            if delta >= t.stride(j) and (stride_match is None or t.stride(j) >= stride_match):
                stride_match = t.stride(j)
                it_dim_map[i] = j
                relative_stride[i] = delta // stride_match
    return it_dim_map, relative_stride, zero_index
```


