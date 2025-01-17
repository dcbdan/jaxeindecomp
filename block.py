from graph import *

def block(headdim = 128, nheads = 32, bsz = 1, seqlen = 1234, hidden = 11008):
  """
  Simplified transformer block
  """

  Input = InputOp

  i3 = Node("i3", Input([nheads, headdim, nheads, headdim]), [])
  i4 = Node("i4", Input([nheads, headdim, nheads, headdim]), [])
  i5 = Node("i5", Input([nheads, headdim, nheads, headdim]), [])
  i7 = Node("i7", Input([bsz, seqlen, nheads, headdim]), [])

  e8 = Node("e8", Contraction(
    "abef,cdef->abcd",
    [bsz, seqlen, nheads, headdim, nheads, headdim]),
    [i7, i3])

  e10 = Node("e10", Contraction(
    "abef,cdef->abcd",
    [bsz, seqlen, nheads, headdim, nheads, headdim]),
    [i7, i4])

  e12 = Node("e12", Contraction(
    "abef,cdef->abcd",
    [bsz, seqlen, nheads, headdim, nheads, headdim]),
    [i7, i5])

  e14 = Node("e14", Contraction(
    "acbe,adbe->abcd",
    [bsz, nheads, seqlen, seqlen, headdim]),
    [e8, e10])

  e16 = Node("e16",
    Scale([bsz, nheads, seqlen, seqlen]),
    [e14])

  e17 = Node("e17",
    Maximum("abcd->abc", [bsz, nheads, seqlen, seqlen]),
    [e16])

  e19 = Node("e19", Subtract(
    "abcd,abc->abcd",
    [bsz, nheads, seqlen, seqlen]),
    [e16, e17])

  e20 = Node("e20", Exp(
    [bsz, nheads, seqlen, seqlen]),
    [e19])

  e21 = Node("e21", Reduction(
    "abcd->abc",
    [bsz, nheads, seqlen, seqlen]),
    [e20])

  e23 = Node("e23", Division(
    "abcd,abc->abcd",
    [bsz, nheads, seqlen, seqlen]),
    [e20, e21])

  e24 = Node("e24", Contraction(
    "abce,aebd->abcd",
    [bsz, nheads, seqlen, headdim, seqlen]),
    [e23, e12])

  i6 = Node("i6", Input([nheads, headdim, nheads, headdim]), [])

  e26 = Node("e26", Contraction(
    "aebf,cdef->abcd",
    [bsz, seqlen, nheads, headdim, nheads, headdim]),
    [e24, i6])

  e28 = Node("e28", Add(
    "abcd,abcd->abcd",
    [bsz, seqlen, nheads, headdim]),
    [i7, e26])

  i0 = Node("i0", Input([hidden, nheads, headdim]), [])
  i2 = Node("i2", Input([hidden, nheads, headdim]), [])

  e29 = Node("e29", Contraction(
    "abde,cde->abc",
    [bsz, seqlen, hidden, nheads, headdim]),
    [e28, i0])


  e32 = Node("e32", Contraction(
    "abde,cde->abc",
    [bsz, seqlen, hidden, nheads, headdim]),
    [e28, i2])

  e31 = Node("e31", Relu([bsz, seqlen, hidden]), [e29])

  e34 = Node("e34", Add("abc,abc->abc", [bsz, seqlen, hidden]), [e31, e32])

  i1 = Node("i1", Input([nheads, headdim, hidden]), [])

  e35 = Node("e35", Contraction(
    "abe,cde->abcd",
    [bsz, seqlen, nheads, headdim, hidden]),
    [e34, i1])

  e37 = Node("e37", Add(
    "abcd,abcd->abcd",
    [bsz, seqlen, nheads, headdim]),
    [e28, e35])

  return e37

from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

def block_input_sharding_split_batch(nlocs):
  # Just data parallel...
  mesh = jax.make_mesh([nlocs], ["i"])
  def make_split(idx, n):
    ret = [None]*n
    ret[idx] = "i"
    return NamedSharding(mesh, P(*ret))
  def make_replicate(n):
    ret = [None]*n
    return NamedSharding(mesh, P(*ret))
  return {
    "i3": make_replicate(4),
    "i4": make_replicate(4),
    "i5": make_replicate(4),
    "i7": make_split(0, 4),
    "i6": make_replicate(4),
    "i0": make_replicate(3),
    "i2": make_replicate(3),
    "i1": make_replicate(3) }

from itertools import product
def block_input_sharding_split_heads(nlocs, allofem = False):
  if nlocs == 1:
    nA, nB = 1, 1
  elif nlocs == 2:
    nA, nB = 2, 1
  elif nlocs == 4:
    nA, nB = 2, 2
  elif nlocs == 8:
    nA, nB = 4, 2
  elif nlocs == 16:
    nA, nB = 4, 4
  elif nlocs == 32:
    nA, nB = 8, 4
  else:
    raise ValueError("invalid sizing")
  mesh = jax.make_mesh([nA, nB], ["i", "j"])
  def make_both(i, j, n):
    ret = [None]*n
    ret[i] = "i"
    ret[j] = "j"
    return NamedSharding(mesh, P(*ret))
  def make_lhs(i, n):
    ret = [None]*n
    ret[i] = "i"
    return NamedSharding(mesh, P(*ret))
  def make_rhs(j, n):
    ret = [None]*n
    ret[j] = "j"
    return NamedSharding(mesh, P(*ret))
  if allofem:
    pass
    #keys = ["i3", "i4", "i5", "i7", "i6", "i0", "i2", "i1"]
    #opts = [
    #          [make_both(0, 2, 4), make_both(2, 0, 4)],
    #          [make_both(0, 2, 4), make_both(2, 0, 4)],
    #          [make_both(0, 2, 4), make_both(2, 0, 4)],
    #          [make_rhs(2, 4), make_lhs(2, 4)],
    #          [make_both(0, 2, 4), make_both(2, 0, 4)],
    #          [make_lhs(1, 3), make_rhs(1, 3)],
    #          [make_lhs(1, 3), make_rhs(1, 3)],
    #          [make_rhs(1, 3), make_lhs(1, 3)]]
    #for opt in product(*opts):
    #  yield {k: v for k, v in zip(keys, opt)}
  else:
    return {
        "i3": make_both(0, 2, 4),
        "i4": make_both(0, 2, 4),
        "i5": make_both(0, 2, 4),
        "i7": make_lhs(2, 4),
        "i6": make_both(2, 0, 4),
        "i0": make_lhs(1, 3),
        "i2": make_lhs(1, 3),
        "i1": make_rhs(1, 3)
    }


