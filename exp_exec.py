import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices

from graph import graph_init_zeros, graph_exec
from block import block
from utils import catchtime

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

#exp01()


