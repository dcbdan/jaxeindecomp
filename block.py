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

