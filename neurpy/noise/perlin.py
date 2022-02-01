import math, torch as th

def perlin_2d(shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = th.stack(th.meshgrid(th.arange(0, res[0], delta[0]), th.arange(0, res[1], delta[1])), dim = -1) % 1
    angles = 2*math.pi*th.rand(res[0]+1, res[1]+1)
    gradients = th.stack((th.cos(angles), th.sin(angles)), dim = -1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
    dot = lambda grad, shift: (th.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * th.lerp(th.lerp(n00, n10, t[..., 0]), th.lerp(n01, n11, t[..., 0]), t[..., 1])


def octaves_2d(shape, octaves=1, persistence=0.5):
    shape_ = th.tensor(shape)
    shape_ = 2 ** th.ceil(th.log2(shape_))
    shape_ = shape_.type(th.int)

    max_octaves = int(min(octaves, math.log(shape_[0]) / math.log(2), math.log(shape_[1]) / math.log(2)))
    res = th.floor(shape_ / 2 ** max_octaves).type(th.int)

    noise = th.zeros(list(shape_))
    frequency, amplitude = 1, 1
    for _ in range(max_octaves):
        noise += amplitude * perlin_2d(shape_, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence

    return noise[:shape[0],:shape[1]]


def random_perlin(shape, amp=0.1, octaves=6, device='cuda'):
    r = octaves_2d(shape, octaves)
    g = octaves_2d(shape, octaves)
    b = octaves_2d(shape, octaves)
    rgb = ( th.stack((r,g,b)) * amp + 1 ) * 0.5
    return rgb.unsqueeze(0).clip(0,1).to(device)
