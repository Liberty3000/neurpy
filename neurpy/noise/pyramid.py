import math, torch as th

def random_pyramid(shape, octaves=5, decay=1.):
    n, c, h, w = shape
    noise = th.zeros([n, c, 1, 1])
    max_octaves = int(min(math.log(h)/math.log(2), math.log(w)/math.log(2)))
    if octaves is not None and 0 < octaves:
      max_octaves = min(octaves, max_octaves)
    for i in reversed(range(int(max_octaves))):
        h_cur, w_cur = h // 2**i, w // 2**i
        noise = th.nn.functional.interpolate(noise, (h_cur, w_cur), mode='bicubic', align_corners=False)
        noise += (th.randn([n, c, h_cur, w_cur]) / max_octaves ) * decay**( max_octaves - (i+1) )
    return noise
