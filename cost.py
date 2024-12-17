from utils import *
from graph import *

# 1. start with a node
# 2. recurse down
# 3. cache (node, partition) pairs

def solve_partitions(output_node: Node, target_n):
  log2_n = get_power_of_2(target_n)
  if log2_n is None:
    raise ValueError("invalid target_n; must be a power of 2")
  ret = None
  best = None
  for p in output_node.op.out_partitions(log2_n):
    all_ps, cost = solve_partitions_(output_node, log2_n, p)
    if best is None or cost < best:
      best = cost
      ret = all_ps
  return all_ps

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
  print(es_shape)
  print(inn_part)
  print(out_part)

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
    print(f"{mode}: {a}, {b}")
    to_out += max(a,b) - a
    to_inn += max(a,b) - b

  print(f"to_out: {2**to_out}")
  print(f"to_inn: {2**to_inn}")
  print(f"inn blksize {inn_blocksize}")
  print(f"out blksize {out_blocksize}")

  cost_to_out   = (2**to_out)       * inn_blocksize * n_inn_blocks
  cost_move_out = ((2**to_inn) - 1) * out_blocksize * n_out_blocks
  print(f"cost to out   {cost_to_out}")
  print(f"cost move out {cost_move_out}")

  return cost_to_out + cost_move_out

def all_input_partitions(node, log2_n):
  options = [inn_node.op.out_partitions(log2_n) for inn_node in node.inputs]
  return itertools_product(*options)

def union_parts(ps):
  if len(ps) == 0:
    return {}

  ret = {k:v for k,v in ps[0].items()}
  for d in ps[1:]:
    for k, v in d.items():
      if k in ret:
        raise ValueError("expect keys to be disjoint!")
      ret[k] = v
  return ret

@cache
def solve_partitions_(node, log2_n, p):
  es_shape = node.op.es_shape()
  if len(p) == node.op.es_rank():
    # this is the join portion
    best = None
    ret = None
    join_cost = cost_join(es_shape, node.op.es_parts(), p)
    fini_parts = node.get_expected_input_parts(p)
    inn_shapes = node.get_inn_shapes()
    for input_parts in all_input_partitions(node, log2_n):
      all_ps = []
      cost = 0
      for init_part, inn_shape, fini_part, inn_node in zip(input_parts, inn_shapes, fini_parts, node.inputs):
        print("A", inn_shape, init_part, fini_part)
        all_ps_, cost_init = solve_partitions_(inn_node, log2_n, init_part)
        all_ps.append(all_ps_)
        cost += cost_init
        cost += cost_repart(inn_shape, init_part, fini_part)
      cost += join_cost

      if best is None or cost < best:
        best = cost
        ret = union_parts(all_ps)

    ret[(node, "Join")] = p
    return ret, best
  else:
    # this is the agg portion
    best = None
    ret = None
    for join_p in node.op.join_partitions(log2_n):
      all_ps, cost = solve_partitions_(node, log2_n, join_p)
      print("B", es_shape, join_p, p)
      cost += cost_repart(es_shape, join_p, p)

      if best is None or cost < best:
        ret = all_ps
        best = cost

    ret[(node, "Agg")] = p
    return ret, best


