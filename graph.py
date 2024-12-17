from utils import *

class BaseOp:
  def es_str(self):
    raise NotImplementedError("BaseOp: es str")

  def es_shape(self):
    raise NotImplementedError("BaseOp: einsum shape")

  def es_modes(self):
    ret = []
    for k in self.es_shape().keys():
      if len(k) != 1 or not k.isalpha():
        raise ValueError("keys must all be single alpha characters")
      ret.append(k)
    return "".join(sorted(ret))

  def out_modes(self):
    return self.es_parts()[-1]

  def es_rank(self):
    return len(self.es_modes())

  def out_rank(self):
    return len(self.out_modes())

  def out_shape(self):
    shape = self.es_shape()
    out_modes = self.es_parts()[-1]
    return [shape[mode] for mode in out_modes]

  def es_parts(self):
    s = self.es_str()
    maybe = s.split("->")
    if len(maybe) == 1:
      # no arrow
      ret = s.split(",")
      if len(ret) != 1:
        raise ValueError("es str: can't have commas without arrow")
      return ret
    elif len(maybe) == 2:
      inns, out = maybe
      inns = inns.split(",")
      return inns + [out]
    else:
      raise ValueError("must only have one arrow")

  def es_agg_modes(self):
    parts = self.es_parts()
    if len(parts) == 1:
      return []
    inns, out = parts[:-1], parts[-1]

    inn_modes = set()
    for inn in inns:
      for mode in inn:
        inn_modes.add(mode)

    out_modes = set()
    for mode in out_modes:
      out_modes.add(mode)

    ret = set()
    for m in inn_modes:
      if m not in out_modes:
        ret.add(m)
    return ret

  def has_aggregation(self):
    return len(self.es_agg_modes()) > 0

  def _all_partitions(self, log2_n, modes):
    for logpart in all_sum_tos(log2_n, len(modes)):
      ret = frozendict({mode: 2**v for mode, v in zip(modes, logpart)})
      if self.is_valid_partition(ret):
        yield ret

  def join_partitions(self, log2_n):
    for x in self._all_partitions(log2_n, self.es_modes()):
      yield x

  def out_partitions(self, log2_n):
    for x in self._all_partitions(log2_n, self.out_modes()):
      yield x

  def is_valid_partition(self, p): # p is a mode -> int mapping
    shape = self.es_shape()
    for mode, part in p.items():
      if part > shape[mode]:
        return False
    return True

class MatmulOp(BaseOp):
  def __init__(self, ni, nj, nk):
    self.ni = ni
    self.nj = nj
    self.nk = nk
  def es_str(self):
    return "ij,jk->ik"
  def es_shape(self):
    return {"i": self.ni, "j": self.nj, "k": self.nk}

class InputOp(BaseOp):
  def __init__(self, shape):
    self.shape = shape
  def es_str(self):
    return "abcdefghijklmnopqrstuvwxyz"[:len(self.shape)]
  def es_shape(self):
    return {mode: sz for mode, sz in zip(self.es_str(), self.shape)}

class Node:
  def __init__(self, name, op, inputs):
    self.name = name
    self.op = op
    for inn in inputs:
        inn.add_output(self)
    self.inputs = list(inputs)
    self.outputs = set()

  def __repr__(self):
    return self.name

  def is_input(self):
    return False

  def add_output(self, other):
    self.outputs.add(other)

  def get_inn_shapes(self):
    return self._get_expected_input_infos(self.op.es_shape())

  def get_expected_input_parts(self, p):
    if len(p) != self.op.es_rank():
      raise ValueError("parts needed for all modes")
    return self._get_expected_input_infos(p)

  def _get_expected_input_infos(self, p):
    if set(p.keys()) != set(self.op.es_shape().keys()):
      raise ValueError("invlaid keys provided")

    inn_parts_there = [inn_node.op.out_modes() for inn_node in self.inputs]
    inn_parts_here  = self.op.es_parts()[:-1]
    ret = []
    for modes_there, modes_here in zip(inn_parts_there, inn_parts_here):
      if len(modes_there) != len(modes_here):
        raise ValueError("uh oh, input output len is not as expected...")
      mapping = {}
      for mt, mh in zip(modes_there, modes_here):
        mapping[mt] = p[mh]
      ret.append(mapping)
    return list(map(frozendict, ret))

