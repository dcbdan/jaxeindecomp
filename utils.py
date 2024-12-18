from functools import cache
from itertools import product as itertools_product

from frozendict import frozendict

def abc_to(n):
  if n > 26:
    raise ValueError("abc_to: n > 26")
  return "abcdefghijklmnopqrstuvwxyz"[:n]

def all_sum_tos(n, v):
  if n == 0:
    return [ [0]*v ]
  if n == 1:
    ret = []
    for i in range(v):
      s = [0]*v
      s[i] = 1
      ret.append(s)
    return ret
  if v == 1:
    return [ [n] ]

  return all_sum_tos_(n, v)

@cache
def all_sum_tos_(n, v):
  ret = []
  for i in range(n+1):
    for s in all_sum_tos(n - i, v - 1):
      ret.append([i] + s)
  return ret

def get_power_of_2(n):
  if n == 1:
    return 0
  if n == 2:
    return 1
  if n % 2 != 0:
    return None
  maybe = get_power_of_2(n // 2)
  if maybe is not None:
    return 1 + maybe
  else:
    return None

def product(xs):
  ret = 1
  for x in xs:
    ret *= x
  return ret

#################################
# https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
from time import perf_counter

class catchtime:
  def __enter__(self):
    self.start = perf_counter()
    return self

  def __exit__(self, type, value, traceback):
    self.time = perf_counter() - self.start
    self.readout = f'Time: {self.time:.3f} seconds'
    print(self.readout)
