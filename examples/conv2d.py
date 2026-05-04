import torch
from torch_spyre._C import get_device_dtype, SpyreTensorLayout

torch.manual_seed(0xAFFE)

x = torch.randn(8, 64, 128, 128, dtype=torch.float16)
y = torch.randn(64, 1, 3, 3, dtype=torch.float16)
z = torch.zeros(64, dtype=torch.float16)

z_dev = z.to("spyre")

x_dev = x.to(
    device_layout=SpyreTensorLayout(
        [8, 128, 1, 128, 64],
        [1048576, 128, -1, 1, 16384],
        get_device_dtype(torch.float16),
    )
)

y_dev = y.to(
    device_layout=SpyreTensorLayout(
        [3, 3, 1, 1, 64],
        [3, 1, -1, -1, 9],
        get_device_dtype(torch.float16),
    ),
)

cpu = torch.conv2d(x, y, groups=64)

aiu = torch.compile(torch.conv2d)(x_dev, y_dev, z_dev, groups=64).cpu()

print(cpu[0, 0])
print(aiu[0, 0])

print(torch.abs(cpu - aiu).amax())
