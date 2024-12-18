from utils import *
from graph import *
from trees import partition_into_trees

# 1. start with a node
# 2. recurse down
# 3. cache (node, partition) pairs

def solve_partitions(output_node: Node, target_n, return_tuple = False):
  log2_n = get_power_of_2(target_n)
  if log2_n is None:
    raise ValueError("invalid target_n; must be a power of 2")

  total_ret = {}
  for tree_root, _ in partition_into_trees(output_node):
    ret = None
    best = None
    for p in tree_root.op.out_partitions(log2_n):
      all_ps, cost = solve_partitions_(total_ret)(tree_root, log2_n, p)
      if best is None or cost < best:
        best = cost
        ret = all_ps
    total_ret = unions([total_ret, ret])
  if return_tuple:
    join_parts = {}
    agg_parts  = {}
    for (node, s), part in total_ret.items():
      if s == "Join":
        join_parts[node.name] = part
      else:
        agg_parts[node.name] = part
    return join_parts, agg_parts
  else:
    return total_ret

def cost_join(es_shape, es_parts, join_part):
  ret = 0
  for inn_part in es_parts[:-1]:
     nelems = 1
     for mode in inn_part:
       nelems *= es_shape[mode]
     n_dup = 1
     for mode, sz in es_shape.items():
       if mode not in inn_part:
         n_dup *= join_part[mode]
     ret += nelems*n_dup
  return ret

# Note: this incorporates aggregtations
def cost_repart(es_shape, inn_part, out_part):
  if inn_part == out_part:
    return 0
  if len(inn_part) > len(out_part):
    # gotta do the aggregation
    nagg = 1
    blocksize = 1
    after_agg_part = {}
    for mode, sz in es_shape.items():
      if mode in out_part:
        blocksize *= sz
        after_agg_part[mode] = inn_part[mode]
      else:
        nagg *= inn_part[mode]
    cost_agg = blocksize*(nagg-1)

    return cost_agg + cost_repart(es_shape, after_agg_part, out_part)

  # ok, inn part and out part have the same modes
  if set(inn_part.keys()) != set(out_part.keys()):
    raise ValueError("expect the modes of inn and out part to be the same")

  def _fix(xs):
    ret = {}
    for k, v in xs.items():
      s = get_power_of_2(v)
      if s is None:
        raise ValueError("parts must be in powers of two")
      ret[k] = s
    return ret

  nelems = product(es_shape[mode] for mode in inn_part.keys())

  n_inn_blocks = product(inn_part.values())
  n_out_blocks = product(out_part.values())
  inn_blocksize = nelems // n_inn_blocks
  out_blocksize = nelems // n_out_blocks

  # and assume all the parts are a power of two and convert to
  # power of two
  inn_part = _fix(inn_part)
  out_part = _fix(out_part)

  to_out = 0
  to_inn = 0
  for mode in inn_part.keys():
    a = inn_part[mode]
    b = out_part[mode]
    to_out += max(a,b) - a
    to_inn += max(a,b) - b

  cost_to_out   = (2**to_out)       * inn_blocksize * n_inn_blocks
  cost_move_out = ((2**to_inn) - 1) * out_blocksize * n_out_blocks

  return cost_to_out + cost_move_out

def all_input_partitions(node, log2_n, node_to_set_part = {}):
  def get(inn_node):
    # make sure to do the agg first here
    if (inn_node, "Agg") in node_to_set_part:
      p = node_to_set_part[(inn_node, "Agg")]
      return [ p ]
    elif (inn_node, "Join") in node_to_set_part:
      p = node_to_set_part[(inn_node, "Join")]
      if set(p.keys()) != set(inn_node.op.out_modes()):
        raise ValueError("this join part isn't an out part")
      return [ p ]
    else:
      return inn_node.op.out_partitions(log2_n)

  options = [get(inn_node) for inn_node in node.inputs]
  return itertools_product(*options)

def unions(ps):
  if len(ps) == 0:
    return {}

  ret = {k:v for k,v in ps[0].items()}
  for d in ps[1:]:
    for k, v in d.items():
      ret[k] = v
  return ret

def disjoint_unions(ps):
  if len(ps) == 0:
    return {}
  for x in ps:
    keys = []
    for k, _ in x.keys():
      keys.append(k.name)

  ret = {k:v for k,v in ps[0].items()}
  for d in ps[1:]:
    for k, v in d.items():
      if k in ret:
        raise ValueError("expect keys to be disjoint! (maybe this was never a tree...)")
      ret[k] = v
  return ret

class solve_partitions_:
  def __init__(self, prev_ps = {}):
    self.prev_ps = prev_ps

  def _in_prev(self, node):
    # (all nodes have an associated join)
    return (node, "Join") in self.prev_ps

  @cache
  def __call__(self, node, log2_n, p):
    es_shape = node.op.es_shape()
    if len(p) == node.op.es_rank():
      # this is the join portion
      best = None
      ret = None
      join_cost = cost_join(es_shape, node.op.es_parts(), p)
      fini_parts = node.get_expected_input_parts(p)
      inn_shapes = node.get_inn_shapes()
      for input_parts in all_input_partitions(node, log2_n, self.prev_ps):
        all_ps = []
        cost = 0
        for init_part, inn_shape, fini_part, inn_node in zip(input_parts, inn_shapes, fini_parts, node.inputs):
          if not self._in_prev(inn_node):
            all_ps_, cost_init = self(inn_node, log2_n, init_part)
            all_ps.append(all_ps_)
            cost += cost_init
          cost += cost_repart(inn_shape, init_part, fini_part)
        cost += join_cost

        if best is None or cost < best:
          best = cost
          ret = disjoint_unions(all_ps)

      ret[(node, "Join")] = p
      return ret, best
    else:
      # this is the agg portion
      best = None
      ret = None
      for join_p in node.op.join_partitions(log2_n):
        all_ps, cost = self(node, log2_n, join_p)
        cost += cost_repart(es_shape, join_p, p)

        if best is None or cost < best:
          ret = all_ps
          best = cost

      ret[(node, "Agg")] = p
      return ret, best


