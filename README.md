
# `steganotorchy`: Hiding messages inside PyTorch model weights

The 32-bit floating-point representation of `1.0` in binary is
```
00111111100000000000000000000000
|`---,--'`---------,-----------'
|    |             `-- 23 bits -- significand
|    `----------------- 8 bits -- exponent
`---------------------- 1 bit --- sign
```
If we change the lowest bit of the significand from `0` to `1`, then the number changes ever so noticeably, from `1.0` to `1.0000001`.

The ASCII encoding of the letter `a` is `0x61`, or `01100001` in binary.
So we can hide `a` inside eight 32-bit floating-point numbers by changing only the lowest bit:
```
00111111100000000000000000000000 -> 1.0
00111111100000000000000000000001 -> 1.0000001
00111111100000000000000000000001 -> 1.0000001
00111111100000000000000000000000 -> 1.0
00111111100000000000000000000000 -> 1.0
00111111100000000000000000000000 -> 1.0
00111111100000000000000000000000 -> 1.0
00111111100000000000000000000001 -> 1.0000001
```

We only need four floating-point numbers if we change the lowest two bits:
```
00111111100000000000000000000001 -> 1.0000001
00111111100000000000000000000010 -> 1.0000002
00111111100000000000000000000000 -> 1.0
00111111100000000000000000000001 -> 1.0000001
```

Or just one floating-point number if we use the lowest eight bits:
```
00111111100000000000000001100001 -> 1.0000116
```

This means that we can hide a 1 KB message inside the weights and biases of any PyTorch model that has at least 1,024 parameters.

