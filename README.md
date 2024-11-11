
# `steganotorchy`: Hiding messages inside neural network weights and biases

`steganotorchy` lets you embed messages inside the weights and biases of your neural network.
It requires that the model be saved in the [safetensors](https://github.com/huggingface/safetensors) format.
(By default, PyTorch saves tensors using Python's pickle.  But you don't want to load pickles that can [execute arbitrary code](https://docs.python.org/3/library/pickle.html), do you?)

## Basic idea

The 32-bit floating-point representation of `1.0` in binary is
```
00111111100000000000000000000000
│╰───┬──╯╰─────────┬───────────╯
│    │             ╰── 23 bits ── significand
│    ╰───────────────── 8 bits ── exponent
╰────────────────────── 1 bit ─── sign
```
If we change the lowest bit of the significand from `0` to `1`, then the number changes ever so noticeably, from `1.0` to `1.0000001`.

The ASCII encoding of the letter `a` is `0x61`, or `01100001` in binary.
So we can hide `a` inside eight 32-bit floating-point numbers by changing only the lowest bit:

| Binary representation              | Float       |
|------------------------------------|-------------|
| `00111111100000000000000000000000` | `1.0`       |
| `00111111100000000000000000000001` | `1.0000001` |
| `00111111100000000000000000000001` | `1.0000001` |
| `00111111100000000000000000000000` | `1.0`       |
| `00111111100000000000000000000000` | `1.0`       |
| `00111111100000000000000000000000` | `1.0`       |
| `00111111100000000000000000000000` | `1.0`       |
| `00111111100000000000000000000001` | `1.0000001` |

We only need four floating-point numbers if we change the lowest two bits:

| Binary representation              | Float       |
|------------------------------------|-------------|
| `00111111100000000000000000000001` | `1.0000001` |
| `00111111100000000000000000000010` | `1.0000002` |
| `00111111100000000000000000000000` | `1.0`       |
| `00111111100000000000000000000001` | `1.0000001` |

Or just one floating-point number if we use the lowest eight bits:

| Binary representation              | Float       |
|------------------------------------|-------------|
| `00111111100000000000000001100001` | `1.0000116` |

This means that we can hide a 1 KB message inside the weights and biases of any neural network that has at least 1,024 parameters.

## Installation

Installing `steganotorchy` requires Cargo.
If you have Cargo, then run:

```
cargo install --git https://github.com/gn0/steganotorchy.git
```

If `$HOME/.cargo/bin` is not in your `PATH` environment variable yet, then you also need to run:

```
export PATH=$HOME/.cargo/bin:$PATH
```

To make this setting permanent:

```
echo 'export PATH=$HOME/.cargo/bin:$PATH' >> $HOME/.bashrc  # If using bash.
echo 'export PATH=$HOME/.cargo/bin:$PATH' >> $HOME/.zshrc   # If using zsh.
```

