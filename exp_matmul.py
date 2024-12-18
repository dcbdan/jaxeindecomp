from setup import *

N = 3000
JOIN_I = 2
JOIN_J = 2
JOIN_K = 2

AGG_I = 4
AGG_K = 2

join_mesh = jax.make_mesh((JOIN_I, JOIN_J, JOIN_K), ('i', 'j', 'k'))
agg_mesh  = jax.make_mesh((AGG_I, AGG_K), ('i', 'k'))

@partial(shard_map, mesh=join_mesh,
  in_specs =(P('i', 'j'), P('j', 'k')),
  out_specs=P('i', 'j', 'k'))
def matmul_join(x_block, y_block):
  z_block = (x_block @ y_block)
  n, m = z_block.shape
  return z_block.reshape(n, 1, m)

x = jax.device_put(jnp.arange(N * N).reshape(N, N), NamedSharding(join_mesh, P('i', 'j')))
y = jax.device_put(jnp.arange(N * N).reshape(N, N), NamedSharding(join_mesh, P('j', 'k')))

z0 = with_sharding_constraint(
  matmul_join(x, y),
  NamedSharding(join_mesh, P('i', 'j', 'k')))

z1 = with_sharding_constraint(z0,
  NamedSharding(agg_mesh, P('i', None, 'k')))

z2 = with_sharding_constraint(
  jnp.einsum("ijk->ik", z1),
  NamedSharding(agg_mesh, P('i', 'k')))

print("X")
print_info(x)

print("\nY")
print_info(y)

print("\nZ0")
print_info(z0)

print("\nZ1")
print_info(z1)

print("\nZ2")
print_info(z2)



