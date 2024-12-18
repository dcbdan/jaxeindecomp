from graph import graph_has_path

#     X
#    A B M
#      C N
#      D O
#     S P
#      Q
#
# As an invariant, force that all nodes
# in a tree must have exactly one path to the root,
# otherwise the node itself becomes another root

def partition_into_trees(root):
  def _has_path_not_through(inn, out, not_here):
    for node in inn.outputs:
      if node != not_here:
        if graph_has_path(node, out):
          return True
    return False

  fst = lambda x: x[0]

  trees = []
  seen = set()
  pending_roots = [root]
  while len(pending_roots) > 0:
    tree_root = pending_roots.pop()

    tree_nodes = set()
    pending = [tree_root]
    while len(pending) > 0:
      node = pending.pop()

      tree_nodes.add(node)

      for inn in node.inputs_as_set():
        if inn not in seen:
          seen.add(inn)

          if _has_path_not_through(inn, tree_root, node):
            pending_roots.append(inn)
          else:
            pending.append(inn)

    trees.append((tree_root, tree_nodes))
  return list(reversed(trees))
