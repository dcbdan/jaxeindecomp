from graph import *
from cost import solve_partitions
from trees import partition_into_trees
from block import block

def exp02():
  X = Node("X", InputOp([20,40]), [])
  Y = Node("Y", InputOp([40,60]), [])
  Z = Node("Z", InputOp([60,80]), [])
  XY = Node("XY", MatmulOp(20, 40, 60), [X, Y])
  YZ = Node("YZ", MatmulOp(40, 60, 80), [XY, Z])

  print(f"X  {X.op.es_parts()}")
  print(f"Y  {Y.op.es_parts()}")
  print(f"Z  {Z.op.es_parts()}")
  print(f"XY {XY.op.es_parts()}")
  print(f"YZ {YZ.op.es_parts()}")

  print(XY.op.es_str())
  print(YZ.op.es_str())
  for p in YZ.op.join_partitions(3):
    print(p)
  for p in YZ.op.out_partitions(3):
    print(p)

  print('================================================')
  for k, v in solve_partitions(YZ, 8).items():
    print(k, v)

def exp03():
  print(cost_repart(
    {'a': 10, 'b': 10},
    {'a': 1, 'b': 2},
    {'a': 1, 'b': 4}))

def exp04():
  X = Node("X", InputOp([20,40]), [])
  Y = Node("Y", InputOp([40,60]), [])
  XY = Node("XY", MatmulOp(20, 40, 60), [X, Y])

  for k, v in solve_partitions(XY, 4).items():
    print(k, v)

def exp05():
  X = block()

  sol = solve_partitions(X, 4)
  print("=====================================")
  for (k, t), v in sol.items():
    print(k.name, t, dict(v))

def exp06():
  out = block()
  all_nodes = graph_all_nodes(out)
  print("num nodes: ", len(all_nodes))
  print("has path from i0 to e37:", graph_has_path(all_nodes["i0"], all_nodes["e37"]))
  print("has path from e29 to e32:", graph_has_path(all_nodes["e29"], all_nodes["e32"]))
  print("-----------------------------------------------")

  for root, tree_nodes in partition_into_trees(out):
    print(root.name, len(tree_nodes))

def exp_exec():

exp05()
