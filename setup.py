import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices

from functools import partial

import jax
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P
from jax.sharding import NamedSharding, PositionalSharding
from jax.experimental.shard_map import shard_map
from jax.lax import with_sharding_constraint
from jax.debug import visualize_sharding, visualize_array_sharding

f_max_diff = lambda x, y: jnp.max(jnp.abs(x - y))

def print_info(x):
  for i, shard in enumerate(x.global_shards):
    print(f"shard {i} @ {str(shard.device)} with shape {shard.data.shape} and index {str(shard.index)}")

