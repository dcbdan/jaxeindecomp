from utils import *

import jax
import jax.numpy as jnp

from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

class BaseOp:
  def es_str(self):
    raise NotImplementedError("BaseOp: es str")

  def es_shape(self):
    raise NotImplementedError("BaseOp: einsum shape")

  def exec(self, *args):
    raise NotImplementedError("BaseOp: exec")

  def exec_decomp(self, *args, **kwargs):
    """
    args must provide the input tensors
    kwargs must provide the join and agg partitions
    """
    raise NotImplementedError("BaseOp: exec decomp")


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
  def exec(self, lhs, rhs):
    return lhs @ rhs
  def exec_decomp(self, x, y, join_part=None, agg_part=None):
    if join_part is None or agg_part is None:
      raise ValueError("join and agg parts must be set for matmul")

    join_mesh = jax.make_mesh([join_part[m] for m in "ijk"], ("i", "j", "k"))
    agg_mesh  = jax.make_mesh([agg_part[m] for m in "ik"], ("i", "k"))

    @partial(shard_map, mesh=join_mesh,
      in_specs =(P('i', 'j'), P('j', 'k')),
      out_specs=P('i', 'j', 'k'))
    def matmul_join(x_block, y_block):
      z_block = x_block * y_block
      m, n = z_block.shape
      return z_block.reshape(m, 1, n)

    z = with_sharding_constraint(
      matmul_join(x, y),
      NamedSharding(join_mesh, P('i', 'j', 'k')))

    z = with_sharding_constraint(z,
      NamedSharding(agg_mesh, P('i', None, 'k')))

    z = with_sharding_constraint(
      jnp.einsum("ijk->ik", z),
      NamedSharding(agg_mesh, P('i', 'k')))

    return z

def _abc_shape(es_str, es_shape):
  modes = sorted(set(c for c in es_str if c.isalpha()))
  if len(modes) != len(es_shape):
    raise ValueError("abc str fail")
  return {k: v for k, v in zip(modes, es_shape)}

class Contraction(BaseOp):
  def __init__(self, abc_str, abc_shape):
    self._es_str   = abc_str
    self._es_shape = _abc_shape(abc_str, abc_shape)
  def es_str(self):
    return self._es_str
  def es_shape(self):
    return self._es_shape
  def exec(self, lhs, rhs):
    return jnp.einsum(self._es_str, lhs, rhs)
  #def exec_decomp(self, x, y, join_part=None, agg_part=None):
  #  if join_part is None or agg_part is None:
  #    raise ValueError("join and agg parts must be set for matmul")
  #  # TODO
  #  join_mesh = jax.make_mesh([join_part[m] for m in "ijk"], ("i", "j", "k"))
  #  agg_mesh  = jax.make_mesh([agg_part[m] for m in "ik"], ("i", "k"))

def _unary_ew_str_shape(shape):
  m = abc_to(len(shape))
  return m + "->" + m, {k: v for k, v in zip(m, shape)}

class Scale(BaseOp):
  def __init__(self, shape):
    self._es_str, self._es_shape = _unary_ew_str_shape(shape)
  def es_str(self):
    return self._es_str
  def es_shape(self):
    return self._es_shape
  def exec(self, inn):
    return inn*0.012345

class Exp(BaseOp):
  def __init__(self, shape):
    self._es_str, self._es_shape = _unary_ew_str_shape(shape)
  def es_str(self):
    return self._es_str
  def es_shape(self):
    return self._es_shape
  def exec(self, inn):
    return jnp.exp(inn)

class Relu(BaseOp):
  def __init__(self, shape):
    self._es_str, self._es_shape = _unary_ew_str_shape(shape)
  def es_str(self):
    return self._es_str
  def es_shape(self):
    return self._es_shape
  def exec(self, inn):
    return jax.nn.relu(inn)

class Maximum(BaseOp):
  def __init__(self, abc_str, abc_shape):
    self._es_str   = abc_str
    self._es_shape = _abc_shape(abc_str, abc_shape)
  def es_str(self):
    return self._es_str
  def es_shape(self):
    return self._es_shape
  def exec(self, inn):
    if self._es_str in ["ab->a", "abc->ab", "abcd->abc", "abcde->abcd"]:
      return jnp.max(inn, axis=-1)
    raise NotImplementedError("Maximum exec op")

class Subtract(BaseOp):
  def __init__(self, abc_str, abc_shape):
    self._es_str   = abc_str
    self._es_shape = _abc_shape(abc_str, abc_shape)
  def es_str(self):
    return self._es_str
  def es_shape(self):
    return self._es_shape
  def exec(self, lhs, rhs):
    if self._es_str == "abcd,abc->abcd":
      new_rhs_shape = rhs.shape + (1,)
      return lhs - rhs.reshape(new_rhs_shape)
    raise NotImplementedError("Subtract exec op: " + self._es_str)

class Reduction(BaseOp):
  def __init__(self, abc_str, abc_shape):
    self._es_str   = abc_str
    self._es_shape = _abc_shape(abc_str, abc_shape)
  def es_str(self):
    return self._es_str
  def es_shape(self):
    return self._es_shape
  def exec(self, inn):
    if self._es_str in ["ab->a", "abc->ab", "abcd->abc", "abcde->abcd"]:
      return jnp.sum(inn, axis=-1)
    raise NotImplementedError("Reduction exec op")

class Division(BaseOp):
  def __init__(self, abc_str, abc_shape):
    self._es_str   = abc_str
    self._es_shape = _abc_shape(abc_str, abc_shape)
  def es_str(self):
    return self._es_str
  def es_shape(self):
    return self._es_shape
  def exec(self, lhs, rhs):
    if self._es_str == "abcd,abc->abcd":
      new_rhs_shape = rhs.shape + (1,)
      return lhs / rhs.reshape(new_rhs_shape)
    raise NotImplementedError("Division exec op")

class Add(BaseOp):
  def __init__(self, abc_str, abc_shape):
    self._es_str   = abc_str
    self._es_shape = _abc_shape(abc_str, abc_shape)
  def es_str(self):
    return self._es_str
  def es_shape(self):
    return self._es_shape
  def exec(self, lhs, rhs):
    parts = self.es_parts()
    if all(p == parts[0] for p in parts[1:]):
      return lhs + rhs
    raise NotImplementedError("Add exec op")

class InputOp(BaseOp):
  def __init__(self, shape):
    self.shape = shape
  def es_str(self):
    return abc_to(len(self.shape))
  def es_shape(self):
    return {mode: sz for mode, sz in zip(self.es_str(), self.shape)}
  def exec(self):
    raise RuntimeError("can't execute input base op...")

class Node:
  def __init__(self, name, op, inputs):
    self.name = name
    self.op = op
    for inn in inputs:
        inn.add_output(self)
    self.inputs = list(inputs)
    self.outputs = set()

  def __repr__(self):
    return "N." + self.name

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

  def inputs_as_set(self):
    for inn in set(self.inputs):
      yield inn

def graph_init_zeros(root):
  return {
    inn.name: jnp.zeros(inn.op.out_shape())
    for inn in graph_all_inputs(root)
  }

def graph_init_zeros_replicated(root, nlocs):
  mesh = jax.make_mesh((nlocs,), ('a',))
  def init(shape):
    return jax.device_put(
      jnp.zeros(shape),
      NamedSharding(mesh, P(*[None for _ in range(len(shape))])))
  return {
    inn.name: init(inn.op.out_shape())
    for inn in graph_all_inputs(root)
  }

def graph_exec(root, data):
  """
  root: the graph node whose tensor is to be computed
  data: mapping from node name to input tensors

  On completion, data contains all intermediate tensors
  """
  def recurse(node):
    if node.name in data:
      return
    for inn in node.inputs:
      recurse(inn)
    data[node.name] = node.op.exec(*[data[inn.name] for inn in node.inputs])

  recurse(root)

def graph_exec_decomp(root, data, join_parts, agg_parts):
  """
  root:  the graph node whose tensor is to be computed
  data:  mapping from node name to input tensors
  join_parts: mapping from node name to join partition
  agg_parts: mapping from node name to agg partition
  """
  def recurse(node):
    if node.name in data:
      return
    for inn in node.inputs:
      recurse(inn)

    args   = [data[inn.name] for inn in node.inputs]

    kwargs = {}
    kwargs["join_part"] = join_parts[node.name]
    if node.op.has_aggregation():
      kwargs["agg_part"] = agg_parts[node.name]

    data[node.name] = node.op.exec_decomp(*args, **kwargs)

  recurse(root)

def graph_all_inputs(root):
  def is_input(node):
    return len(node.inputs) == 0
  return [node for node in graph_all_nodes(root).values() if is_input(node)]

def graph_all_nodes(node):
  """
  return mapping from node.name to the node
  """
  ret = {}
  pending = [node]
  while len(pending) > 0:
    node = pending.pop()
    if node.name in ret:
      continue

    ret[node.name] = node

    pending += list(node.outputs)
    pending += list(node.inputs)

  return ret

def graph_has_path(inn, out):
  seen = set()
  pending = [inn]
  while len(pending) > 0:
    node = pending.pop()
    for mid in node.outputs:
      if mid == out:
        return True
      elif mid not in seen:
        seen.add(mid)
        pending.append(mid)
  return False


