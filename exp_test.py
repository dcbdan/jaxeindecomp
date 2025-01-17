import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices

import graph as g
from graph import graph_all_nodes
from graph import graph_init_zeros, graph_init_zeros_replicated, graph_init_zeros_sharded
from graph import graph_exec, graph_exec_decomp

from block import block
from block import block_input_sharding_split_batch
from block import block_input_sharding_split_heads
from utils import catchtime
from utils import print_sharded_tensor_info as print_info
from cost import solve_partitions

import jax
import jax.numpy as jnp

from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

def exp_with_shards(shards, out):
  @jax.jit
  def exec(tensors):
    graph_exec(out, tensors)
    return tensors[out.name]

  init_data = graph_init_zeros_sharded(out, shards)
  for tensor in init_data.values():
    tensor.block_until_ready()

  with catchtime():
    t = exec({k: v for k, v in init_data.items()})
    t.block_until_ready()
  with catchtime() as timer:
    t = exec(init_data)
    t.block_until_ready()
  return timer.time

def test(model, bsz, slen, nlocs):
  if model == "7B":
    out = block(headdim=128,nheads=32,hidden=11008,bsz=bsz,seqlen=slen)
  elif model == "13B":
    out = block(headdim=128,nheads=40,hidden=13824,bsz=bsz,seqlen=slen)
  elif model == "30B":
    out = block(headdim=128,nheads=52,hidden=17920,bsz=bsz,seqlen=slen)
  elif model == "65B":
    out = block(headdim=128,nheads=64,hidden=22016,bsz=bsz,seqlen=slen)
  else:
    raise ValueError("invalid model")

  join_parts, agg_parts = solve_partitions(out, nlocs, return_tuple=True)

  def exec_jaxeindecomp():
    print("exec jax eindecomp!")
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
    with catchtime() as timer:
      t = exec(init_data)
      t.block_until_ready()
    return timer.time

  def exec_split_batch():
    print("exec split batch!")
    if bsz < nlocs:
      return "NA"
    shards = block_input_sharding_split_batch(nlocs)
    return exp_with_shards(shards, out)
  def exec_split_heads():
    print("exec split heads!")
    shards = block_input_sharding_split_heads(nlocs)
    return exp_with_shards(shards, out)

  t_decomp = exec_jaxeindecomp()
  t_batch  = exec_split_batch()
  t_heads  = exec_split_heads()
  if t_batch == "NA":
    timestr = f"decomp={t_decomp:.5f}, batch={t_batch}, heads={t_heads:.5f}"
  else:
    timestr = f"decomp={t_decomp:.5f}, batch={t_batch:.5f}, heads={t_heads:.5f}"
  print(f"RESULT: {model}, bsz={bsz}, slen={slen}, nlocs={nlocs}: " + timestr, flush=True)

test("7B", 1, 1000, 8)
test("7B", 2, 1000, 8)
test("7B", 4, 1000, 8)
test("7B", 8, 1000, 8)
