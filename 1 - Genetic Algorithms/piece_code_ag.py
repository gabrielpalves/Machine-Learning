L = 4 * 8 # size of chromossome in bits

import struct

def floatToBits(f):
    s = struct.pack('>f', f)
    return struct.unpack('>L', s)[0]

def bitsToFloat(b):
    s = struct.pack('>L', b)
    return struct.unpack('>f', s)[0]

# Exemplo:  1.23 -> '00010111100'
def get_bits(x):
    x = floatToBits(x)
    N = 4 * 8
    bits = ''
    for bit in range(N):
        b = x & (2**bit)
        bits += '1' if b > 0 else '0'
    return bits

# Exemplo:  '00010111100' ->  1.23
def get_float(bits):
    x = 0
    assert(len(bits) == L)
    for i, bit in enumerate(bits):
        bit = int(bit)  # 0 or 1
        x += bit * (2**i)
    return bitsToFloat(x)

