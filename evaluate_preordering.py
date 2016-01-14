#!/usr/bin/python

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A script to evaluate word-preordering performance."""

import sys


def _LessOrEqual(lhs, rhs):
  """Less-or-equal relation of source-side tokens.
  The definition is from ``Neubig et al.: Inducing a Discriminative Parser to
  Optimize Machine Translation Reordering''."""
  return min(lhs) <= min(rhs) and max(lhs) <= max(rhs)


def _ReadAlign(fo):
  """Read one example from the alignment file."""
  line = fo.readline()
  if not line:
    return None
  line = line[:-1]
  fields = line.split()
  if len(fields) < 3:
    sys.exit('Too few fields.')
  if fields[1] != '|||':
    sys.exit('Wrong format.')
  values = fields[0].split('-')
  src_num = int(values[0])
  trg_num = int(values[1])

  # aligns[i] contains the indices of the target tokens which are aligned to
  # the (i+1)-th source token.
  aligns = [set() for _ in range(src_num)]
  for field in fields[2:]:
    values = field.split('-')
    src_id = int(values[0])
    trg_id = int(values[1])
    if src_id < 0 or src_id >= src_num or trg_id < 0 or trg_id >= trg_num:
      sys.exit('Wrong alignment data: %s', line)
    aligns[src_id].add(trg_id)

  sorted_list = []
  for i in range(src_num):
    if not aligns[i]:
      continue
    pos = 0
    eq = False
    while pos < len(sorted_list):
      le = _LessOrEqual(aligns[i], aligns[sorted_list[pos][0]])
      ge = _LessOrEqual(aligns[sorted_list[pos][0]], aligns[i])
      eq = le and ge
      if not le and not ge:
        return []
      if le:
        break
      pos += 1
    if not eq:
      sorted_list.insert(pos, [])
    sorted_list[pos].append(i)
  alignment = [-1] * src_num
  for i in range(len(sorted_list)):
    for j in sorted_list[i]:
      alignment[j] = i
  alignment.append(len(sorted_list))
  return alignment


def _ReadOrder(fo):
  """Read one example from the order file."""
  line = fo.readline()
  if not line:
    return None
  line = line[:-1]
  order = line.split()
  order = [int(item) for item in order]
  return order


def _CalculateTau(alignment, order):
  """Calculate Kendall's Tau."""
  src_num = len(order)
  if src_num <= 1:
    return 1.0
  errors = 0
  for i in range(src_num - 1):
    for j in range(i + 1, src_num):
      if alignment[order[i]] > alignment[order[j]]:
        errors += 1
  tau = 1.0 - float(errors) / (src_num * (src_num - 1) / 2)
  return tau


def _CalculateFRS(alignment, order):
  """Calculate the fuzzy reordering score."""
  src_num = len(order)
  if src_num <= 1:
    return 1.0
  discont = 0
  for i in range(src_num + 1):
    trg_prv = alignment[order[i - 1]] if i - 1 >= 0 else -1
    trg_cur = alignment[order[i]] if i < src_num else alignment[-1]
    if trg_prv != trg_cur and trg_prv + 1 != trg_cur:
      discont += 1
  frs = 1.0 - float(discont) / (src_num + 1)
  return frs


def main(argv):
  if len(argv) != 3:
    print >>sys.stderr, 'usage: %s <align file> <order file>' % argv[0]
    print >>sys.stderr, '    <align file> : Reference data of word preordering.'
    print >>sys.stderr, '    <order file> : Output of word preordering system.'
    sys.exit(1)
  align_file = argv[1]
  order_file = argv[2]

  align_fo = open(align_file)
  order_fo = open(order_file)

  num = 0
  skipped = 0
  frs_sum = 0.0
  tau_sum = 0.0
  cm_sum = 0
  while True:
    alignment = _ReadAlign(align_fo)
    order = _ReadOrder(order_fo)
    if alignment is None and order is None:
      break
    if alignment is None:
      sys.exit('Cannot read the alignment file.')
    if order is None:
      sys.exit('Cannot read the order file.')

    if not alignment:
      skipped += 1
      continue
    if len(alignment) - 1 != len(order):
      sys.exit('Numbers of tokens mismatch.')

    # Remove unaligned tokens.
    for i, a in enumerate(alignment):
      if a < 0:
        order.remove(i)

    num += 1
    frs = _CalculateFRS(alignment, order)
    tau = _CalculateTau(alignment, order)
    frs_sum += frs
    tau_sum += tau
    cm_sum += 1.0 if tau == 1.0 else 0.0

  tau_ave = 0.0 if num == 0 else tau_sum / num
  frs_ave = 0.0 if num == 0 else frs_sum / num
  cm_ave = 0.0 if num == 0 else cm_sum / num
  print 'Number of evaluated sentences: %d' % num
  print 'Number of skipped sentences: %d' % skipped
  print 'Fuzzy Reordering Score: %f' % frs_ave
  print 'Kendall\'s Tau: %f' % tau_ave
  print 'Complete Match: %f' % cm_ave

if __name__ == '__main__':
  main(sys.argv)
