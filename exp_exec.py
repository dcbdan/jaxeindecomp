import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices

import graph as g
from graph import graph_init_zeros, graph_init_zeros_replicated, graph_all_nodes
from graph import graph_exec, graph_exec_decomp

from block import block
from utils import catchtime
from utils import print_sharded_tensor_info as print_info
from cost import solve_partitions

import jax
import jax.numpy as jnp

from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def exp01():
  out = block()

  @jax.jit
  def exec(tensors):
    graph_exec(out, tensors)
    return tensors[out.name]

  d = graph_init_zeros(out)
  for tensor in d.values():
    tensor.block_until_ready()

  with catchtime():
    t = exec({k: v for k, v in d.items()})
    t.block_until_ready()
  with catchtime():
    t = exec(d)
    t.block_until_ready()

def exp02(with_matmul_op = True):
  X = g.Node("X", g.InputOp([1000, 2000]), [])
  Y = g.Node("Y", g.InputOp([2000, 3000]), [])
  Z = g.Node("Z", g.InputOp([3000, 4000]), [])

  if with_matmul_op:
    XY = g.Node("XY", g.MatmulOp(1000, 2000, 3000), [X, Y])
    Out = g.Node("Out",  g.MatmulOp(1000, 3000, 4000), [XY, Z])
  else:
    XY = g.Node("XY", g.Contraction("ac,cb->ab", [1000, 3000, 2000]), [X, Y])
    Out = g.Node("Out", g.Contraction("ac,cb->ab", [1000, 4000, 3000]), [XY, Z])

  nlocs = 8

  join_parts, agg_parts = solve_partitions(Out, nlocs, return_tuple=True)
  for name in graph_all_nodes(Out).keys():
    print(name, "join", join_parts[name])
    if name in agg_parts:
      print(name, "agg ", agg_parts[name])

  @jax.jit
  def exec(tensors):
    graph_exec_decomp(Out, tensors, join_parts, agg_parts)
    return tensors[Out.name]

  init_data = graph_init_zeros_replicated(Out, nlocs)
  for tensor in init_data.values():
    tensor.block_until_ready()

  with catchtime():
    t = exec({k: v for k, v in init_data.items()})
    t.block_until_ready()
  print_info(t)

  with catchtime():
    t = exec(init_data)
    t.block_until_ready()
  print_info(t)

def exp03():
  out = block()

  nlocs = 8

  join_parts, agg_parts = solve_partitions(out, nlocs, return_tuple=True)
  for name in graph_all_nodes(out).keys():
    print(name, "join", join_parts[name])
    if name in agg_parts:
      print(name, "agg ", agg_parts[name])

  @jax.jit
  def exec(tensors):
    graph_exec_decomp(out, tensors, join_parts, agg_parts)
    return tensors[out.name]

  init_data = graph_init_zeros_replicated(out, nlocs)
  for tensor in init_data.values():
    tensor.block_until_ready()

  with catchtime():
    t = exec({k: v for k, v in init_data.items()})
    t.block_until_ready()
  print_info(t)

  with catchtime():
    t = exec(init_data)
    t.block_until_ready()
  print_info(t)

  print(out.op.es_str())
  print(join_parts[out.name])
  if out.op.has_aggregation():
    print(agg_parts[out.name])

exp03()
