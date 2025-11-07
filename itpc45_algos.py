
"""
ITPC-45 (Advanced Data Structure and Algorithms Lab)
Single-file reference implementations.

Everything is written as small, importable functions or self-contained classes.
No external packages are required; only Python's standard library is used.

NOTE:
- These are educational/minimal implementations intended to be clear and correct.
- Time/space complexities are provided in docstrings where helpful.
- Heaps are represented as 0-indexed arrays (lists). For some routines we also
  provide thin wrappers around `heapq` from the stdlib.

Author: ChatGPT (GPT-5 Thinking)
"""

from __future__ import annotations

import math
import random
import itertools
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional, Iterable, Dict, Set

# -----------------------------------------------------------------------------
# 1. Binary Search Tree (BST) with utilities
# -----------------------------------------------------------------------------

class BSTNode:
    __slots__ = ("key", "left", "right")
    def __init__(self, key):
        self.key = key
        self.left: Optional['BSTNode'] = None
        self.right: Optional['BSTNode'] = None

class BST:
    """Simple Binary Search Tree (no self-balancing)."""

    def __init__(self, values: Iterable[Any] = ()):
        self.root: Optional[BSTNode] = None
        for v in values:
            self.insert(v)

    def insert(self, key: Any) -> None:
        """Insert a key into the BST. Average O(log n), worst O(n)."""
        if self.root is None:
            self.root = BSTNode(key)
            return
        cur = self.root
        while True:
            if key < cur.key:
                if cur.left is None:
                    cur.left = BSTNode(key)
                    return
                cur = cur.left
            elif key > cur.key:
                if cur.right is None:
                    cur.right = BSTNode(key)
                    return
                cur = cur.right
            else:
                # ignore duplicates (or you can choose a side consistently)
                return

    def search(self, key: Any) -> bool:
        """Return True iff key is in the tree. Average O(log n), worst O(n)."""
        cur = self.root
        while cur:
            if key < cur.key:
                cur = cur.left
            elif key > cur.key:
                cur = cur.right
            else:
                return True
        return False

    def find_min(self) -> Optional[Any]:
        """Return minimum key in the BST, or None if empty. O(h)."""
        cur = self.root
        if not cur:
            return None
        while cur.left:
            cur = cur.left
        return cur.key

    def height(self) -> int:
        """Return the number of nodes in the longest path (height in nodes)."""
        def h(node: Optional[BSTNode]) -> int:
            if not node:
                return 0
            return 1 + max(h(node.left), h(node.right))
        return h(self.root)

    def mirror(self) -> None:
        """Swap left<->right pointers recursively (tree mirror). O(n)."""
        def rec(node: Optional[BSTNode]) -> None:
            if not node:
                return
            node.left, node.right = node.right, node.left
            rec(node.left)
            rec(node.right)
        rec(self.root)


# -----------------------------------------------------------------------------
# 2. Optimal BST from sorted keys with access probabilities
# -----------------------------------------------------------------------------

def optimal_bst_cost(keys: List[Any], p: List[float]) -> Tuple[float, List[List[int]]]:
    """
    Dynamic Programming O(n^3) for Optimal BST.
    keys must be sorted. p[i] is search probability of keys[i].
    Returns (min_expected_cost, root_table) where root_table[i][j] is the index
    of the root chosen for keys[i..j].
    """
    n = len(keys)
    assert len(p) == n and n > 0, "Length mismatch or empty input."
    # cost[i][j]: min cost for keys i..j inclusive
    cost = [[0.0]*n for _ in range(n)]
    root = [[-1]*n for _ in range(n)]
    # prefix sums of probabilities to get sum p[i..j] in O(1)
    ps = [0.0]
    for x in p:
        ps.append(ps[-1] + x)
    def prob(i, j):  # inclusive
        return ps[j+1] - ps[i]

    for i in range(n):
        cost[i][i] = p[i]
        root[i][i] = i

    for L in range(2, n+1):          # chain length
        for i in range(0, n-L+1):
            j = i + L - 1
            cost[i][j] = float('inf')
            # try each key as root
            for r in range(i, j+1):
                left = cost[i][r-1] if r > i else 0.0
                right = cost[r+1][j] if r < j else 0.0
                c = left + right + prob(i, j)
                if c < cost[i][j]:
                    cost[i][j] = c
                    root[i][j] = r
    return cost[0][n-1], root

def build_optimal_bst(keys: List[Any], root_table: List[List[int]]) -> BST:
    """Build actual BST from a root table produced by optimal_bst_cost."""
    n = len(keys)
    if n == 0:
        return BST()
    nodes = [BSTNode(k) for k in keys]
    def build(i, j) -> Optional[BSTNode]:
        if i > j: return None
        r = root_table[i][j]
        node = nodes[r]
        node.left = build(i, r-1)
        node.right = build(r+1, j)
        return node
    t = BST()
    t.root = build(0, n-1)
    return t

# -----------------------------------------------------------------------------
# 3. Linear Search & Binary Search
# -----------------------------------------------------------------------------

def linear_search(arr: List[Any], target: Any) -> int:
    """
    Return index of target or -1.
    Time: O(n). Space: O(1).
    """
    for i, x in enumerate(arr):
        if x == target:
            return i
    return -1

def binary_search(arr: List[Any], target: Any) -> int:
    """
    Return index of target in sorted array or -1.
    Time: O(log n). Space: O(1).
    """
    lo, hi = 0, len(arr)-1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# -----------------------------------------------------------------------------
# 4,5,7. Quick Sort variants
# -----------------------------------------------------------------------------

def quicksort(arr: List[Any], reverse: bool=False) -> List[Any]:
    """In-place QuickSort (three-way) to handle duplicates. Average O(n log n)."""
    a = list(arr)
    def _qs(lo, hi):
        if lo >= hi: return
        # median-of-three for stability
        mid = (lo + hi)//2
        pivot = sorted([a[lo], a[mid], a[hi]])[1]
        i, j, k = lo, lo, hi
        while j <= k:
            if (a[j] < pivot) ^ reverse:
                a[i], a[j] = a[j], a[i]
                i += 1; j += 1
            elif a[j] == pivot:
                j += 1
            else:
                a[j], a[k] = a[k], a[j]
                k -= 1
        _qs(lo, i-1); _qs(j, hi)
    _qs(0, len(a)-1)
    return a

def quicksort_pivot(arr: List[Any], pivot_mode: str="first") -> List[Any]:
    """QuickSort with explicit pivot choices: 'first'|'last'|'random'."""
    a = list(arr)
    def pick(lo, hi):
        if pivot_mode == "first": return lo
        if pivot_mode == "last": return hi
        return random.randint(lo, hi)
    def part(lo, hi):
        p = pick(lo, hi)
        a[lo], a[p] = a[p], a[lo]
        pivot = a[lo]
        i = lo + 1
        for j in range(lo+1, hi+1):
            if a[j] < pivot:
                a[i], a[j] = a[j], a[i]
                i += 1
        a[lo], a[i-1] = a[i-1], a[lo]
        return i-1
    def _qs(lo, hi):
        if lo < hi:
            p = part(lo, hi)
            _qs(lo, p-1); _qs(p+1, hi)
    _qs(0, len(a)-1)
    return a

# -----------------------------------------------------------------------------
# 6,8,9,10,11. Heaps: building, kth order, priority queue, delete kth, heap sort
# -----------------------------------------------------------------------------

def heap_parent(i): return (i-1)//2
def heap_left(i): return 2*i + 1
def heap_right(i): return 2*i + 2

def heapify_down_min(a: List[Any], i: int) -> None:
    n = len(a)
    while True:
        l, r = heap_left(i), heap_right(i)
        smallest = i
        if l < n and a[l] < a[smallest]: smallest = l
        if r < n and a[r] < a[smallest]: smallest = r
        if smallest == i: break
        a[i], a[smallest] = a[smallest], a[i]
        i = smallest

def heapify_up_min(a: List[Any], i: int) -> None:
    while i > 0 and a[heap_parent(i)] > a[i]:
        p = heap_parent(i)
        a[p], a[i] = a[i], a[p]
        i = p

def build_min_heap(a: List[Any]) -> List[Any]:
    a = list(a)
    for i in range(len(a)//2 -1, -1, -1):
        heapify_down_min(a, i)
    return a

def extract_min(a: List[Any]) -> Any:
    assert a, "Heap underflow"
    a[0], a[-1] = a[-1], a[0]
    v = a.pop()
    if a: heapify_down_min(a, 0)
    return v

def insert_min(a: List[Any], key: Any) -> None:
    a.append(key)
    heapify_up_min(a, len(a)-1)

def kth_min_in_minheap(a: List[Any], k: int) -> Any:
    """Return k-th smallest by popping k-1 items (O(k log n))."""
    h = list(a)
    for _ in range(k-1): extract_min(h)
    return h[0]

def build_max_heap(a: List[Any]) -> List[Any]:
    a = [-x for x in a]
    a = build_min_heap(a)
    return [-x for x in a]

def extract_max(a: List[Any]) -> Any:
    """Assumes max-heap represented by negatives of a min-heap technique for brevity."""
    a0 = [-x for x in a]
    v = extract_min(a0)
    v = -v
    # rebuild a from a0
    a[:] = [-x for x in a0]
    return v

def kth_max_in_maxheap(a: List[Any], k: int) -> Any:
    h = list(a)
    for _ in range(k-1): extract_max(h)
    return h[0]

def heap_sort(a: List[Any]) -> List[Any]:
    """Heap sort using min-heap. O(n log n)."""
    h = build_min_heap(a)
    out = []
    while h:
        out.append(extract_min(h))
    return out

class PriorityQueue:
    """Min-priority queue using our min-heap helpers."""
    def __init__(self, items: Iterable[Any] = ()):
        self.h = build_min_heap(list(items))
    def push(self, x): insert_min(self.h, x)
    def pop(self): return extract_min(self.h)
    def peek(self): return self.h[0] if self.h else None
    def __len__(self): return len(self.h)

def analyze_marks_with_heap(marks: List[int]) -> Tuple[int, int, str]:
    """
    Return (min_mark, max_mark, complexity_note).
    Uses heap for min and max extraction.
    """
    if not marks:
        return (None, None, "No data")
    min_heap = build_min_heap(marks)
    max_heap = build_max_heap(marks)
    mn = min_heap[0]
    mx = max_heap[0]
    note = "Building heap O(n); querying min/max O(1)."
    return (mn, mx, note)

def delete_kth_index_minheap(a: List[Any], k: int) -> None:
    """Delete element at index k (0-based) from a min-heap."""
    n = len(a)
    if not (0 <= k < n): return
    a[k], a[-1] = a[-1], a[k]
    a.pop()
    if k < len(a):
        heapify_up_min(a, k)
        heapify_down_min(a, k)

def delete_kth_index_maxheap(a: List[Any], k: int) -> None:
    """Max-heap variant implemented via sign flip for brevity."""
    if not a: return
    a0 = [-x for x in a]
    delete_kth_index_minheap(a0, k)
    a[:] = [-x for x in a0]

# -----------------------------------------------------------------------------
# 12. Insertion Sort with simple op count
# -----------------------------------------------------------------------------

def insertion_sort(a: List[Any]) -> Tuple[List[Any], int]:
    """
    Insertion sort.
    Returns (sorted_list, number_of_swaps).
    Time: O(n^2) worst, O(n) best (already sorted).
    """
    a = list(a)
    swaps = 0
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j+1] = a[j]
            j -= 1
            swaps += 1
        a[j+1] = key
    return a, swaps

# -----------------------------------------------------------------------------
# 13. AVL Tree (insert, delete, search)
# -----------------------------------------------------------------------------

class AVLNode:
    __slots__ = ("key","left","right","h")
    def __init__(self, key):
        self.key=key; self.left=None; self.right=None; self.h=1

def _avl_h(n: Optional[AVLNode]): return n.h if n else 0
def _avl_upd(n: AVLNode): n.h = 1 + max(_avl_h(n.left), _avl_h(n.right))
def _avl_bal(n: AVLNode): return _avl_h(n.left) - _avl_h(n.right)

def _avl_rot_right(y: AVLNode) -> AVLNode:
    x = y.left; T2 = x.right
    x.right = y; y.left = T2
    _avl_upd(y); _avl_upd(x); return x

def _avl_rot_left(x: AVLNode) -> AVLNode:
    y = x.right; T2 = y.left
    y.left = x; x.right = T2
    _avl_upd(x); _avl_upd(y); return y

class AVLTree:
    def __init__(self): self.root=None

    def search(self, key): 
        n=self.root
        while n:
            if key<n.key: n=n.left
            elif key>n.key: n=n.right
            else: return True
        return False

    def _insert(self, n: Optional[AVLNode], key) -> AVLNode:
        if not n: return AVLNode(key)
        if key < n.key: n.left = self._insert(n.left, key)
        elif key > n.key: n.right = self._insert(n.right, key)
        else: return n
        _avl_upd(n)
        b = _avl_bal(n)
        # LL
        if b>1 and key<n.left.key: return _avl_rot_right(n)
        # RR
        if b<-1 and key>n.right.key: return _avl_rot_left(n)
        # LR
        if b>1 and key>n.left.key:
            n.left = _avl_rot_left(n.left); return _avl_rot_right(n)
        # RL
        if b<-1 and key<n.right.key:
            n.right = _avl_rot_right(n.right); return _avl_rot_left(n)
        return n

    def insert(self, key): self.root = self._insert(self.root, key)

    def _min_node(self, n): 
        while n.left: n=n.left
        return n

    def _delete(self, n: Optional[AVLNode], key) -> Optional[AVLNode]:
        if not n: return None
        if key < n.key: n.left = self._delete(n.left, key)
        elif key > n.key: n.right = self._delete(n.right, key)
        else:
            # remove this node
            if not n.left: return n.right
            if not n.right: return n.left
            t = self._min_node(n.right)
            n.key = t.key
            n.right = self._delete(n.right, t.key)
        _avl_upd(n)
        b=_avl_bal(n)
        if b>1 and _avl_bal(n.left)>=0: return _avl_rot_right(n)
        if b>1 and _avl_bal(n.left)<0:
            n.left=_avl_rot_left(n.left); return _avl_rot_right(n)
        if b<-1 and _avl_bal(n.right)<=0: return _avl_rot_left(n)
        if b<-1 and _avl_bal(n.right)>0:
            n.right=_avl_rot_right(n.right); return _avl_rot_left(n)
        return n

    def delete(self, key): self.root = self._delete(self.root, key)

# -----------------------------------------------------------------------------
# 14. Red-Black Tree (insert/search; deletion omitted for brevity)
# -----------------------------------------------------------------------------

RED, BLACK = True, False

class RBNode:
    __slots__ = ("key","left","right","parent","color")
    def __init__(self, key, color=RED):
        self.key=key; self.left=None; self.right=None; self.parent=None; self.color=color

class RBTree:
    """Classic CLRS-style Red-Black Tree with insert & search (deletion omitted)."""
    def __init__(self):
        self.nil = RBNode(None, color=BLACK)  # sentinel
        self.root = self.nil

    def search(self, key):
        n = self.root
        while n is not self.nil:
            if key<n.key: n=n.left
            elif key>n.key: n=n.right
            else: return True
        return False

    def _left_rotate(self, x: RBNode):
        y = x.right
        x.right = y.left
        if y.left is not self.nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is self.nil:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, y: RBNode):
        x = y.left
        y.left = x.right
        if x.right is not self.nil:
            x.right.parent = y
        x.parent = y.parent
        if y.parent is self.nil:
            self.root = x
        elif y is y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        x.right = y
        y.parent = x

    def insert(self, key):
        z = RBNode(key, color=RED)
        z.left = z.right = self.nil
        y = self.nil
        x = self.root
        while x is not self.nil:
            y = x
            if z.key < x.key: x = x.left
            elif z.key > x.key: x = x.right
            else: return  # ignore duplicates
        z.parent = y
        if y is self.nil: self.root = z
        elif z.key < y.key: y.left = z
        else: y.right = z
        self._insert_fixup(z)

    def _insert_fixup(self, z: RBNode):
        while z.parent.color == RED:
            if z.parent is z.parent.parent.left:
                y = z.parent.parent.right
                if y.color == RED:  # Case 1
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.right:  # Case 2
                        z = z.parent
                        self._left_rotate(z)
                    z.parent.color = BLACK      # Case 3
                    z.parent.parent.color = RED
                    self._right_rotate(z.parent.parent)
            else:
                y = z.parent.parent.left
                if y.color == RED:
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._right_rotate(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self._left_rotate(z.parent.parent)
        self.root.color = BLACK

# -----------------------------------------------------------------------------
# 15. B-Tree (minimum degree t >= 2), insert & search
# -----------------------------------------------------------------------------

class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t=t
        self.leaf=leaf
        self.keys=[]  # len in [t-1 .. 2t-1]
        self.children=[]  # len = len(keys)+1

    def search(self, k):
        i = 0
        while i < len(self.keys) and k > self.keys[i]:
            i += 1
        if i < len(self.keys) and self.keys[i] == k:
            return True
        if self.leaf:
            return False
        return self.children[i].search(k)

    def split_child(self, i):
        t=self.t
        y=self.children[i]
        z=BTreeNode(t, leaf=y.leaf)
        mid = y.keys[t-1]
        z.keys = y.keys[t:]
        y.keys = y.keys[:t-1]
        if not y.leaf:
            z.children = y.children[t:]
            y.children = y.children[:t]
        self.children.insert(i+1, z)
        self.keys.insert(i, mid)

    def insert_nonfull(self, k):
        i = len(self.keys) - 1
        if self.leaf:
            self.keys.append(None)
            while i >= 0 and k < self.keys[i]:
                self.keys[i+1] = self.keys[i]; i -= 1
            self.keys[i+1] = k
        else:
            while i >= 0 and k < self.keys[i]:
                i -= 1
            i += 1
            if len(self.children[i].keys) == 2*self.t - 1:
                self.split_child(i)
                if k > self.keys[i]:
                    i += 1
            self.children[i].insert_nonfull(k)

class BTree:
    def __init__(self, t=2):
        assert t>=2, "t (min degree) must be >= 2"
        self.t=t
        self.root=BTreeNode(t, leaf=True)
    def search(self, k): return self.root.search(k)
    def insert(self, k):
        r=self.root
        if len(r.keys) == 2*self.t - 1:
            s=BTreeNode(self.t, leaf=False)
            s.children=[r]
            s.split_child(0)
            self.root=s
            self._insert_nonfull(s, k)
        else:
            self._insert_nonfull(r, k)
    def _insert_nonfull(self, x: BTreeNode, k):
        x.insert_nonfull(k)

# -----------------------------------------------------------------------------
# 16,17,18. String pattern matching: Naive, Rabin-Karp, KMP
# -----------------------------------------------------------------------------

def naive_search(text: str, pattern: str) -> List[int]:
    """Return list of start indices where pattern occurs in text. O(n*m)."""
    n, m = len(text), len(pattern)
    if m == 0: return list(range(n+1))
    res = []
    for i in range(n-m+1):
        if text[i:i+m] == pattern:
            res.append(i)
    return res

def rabin_karp(text: str, pattern: str, base: int=256, mod: int=10**9+7) -> List[int]:
    """Rolling-hash string search. Expected O(n+m)."""
    n, m = len(text), len(pattern)
    if m == 0: return list(range(n+1))
    res = []
    h = pow(base, m-1, mod)
    ph = 0
    th = 0
    for i in range(m):
        ph = (ph*base + ord(pattern[i])) % mod
        th = (th*base + ord(text[i])) % mod
    for i in range(n-m+1):
        if ph == th and text[i:i+m] == pattern:
            res.append(i)
        if i < n-m:
            th = ((th - ord(text[i])*h) * base + ord(text[i+m])) % mod
            if th < 0: th += mod
    return res

def kmp_build_lps(pattern: str) -> List[int]:
    lps=[0]*len(pattern)
    j=0
    for i in range(1, len(pattern)):
        while j>0 and pattern[i]!=pattern[j]:
            j=lps[j-1]
        if pattern[i]==pattern[j]:
            j+=1; lps[i]=j
    return lps

def kmp(text: str, pattern: str) -> List[int]:
    """Knuth-Morris-Pratt string search. O(n+m)."""
    n, m = len(text), len(pattern)
    if m == 0: return list(range(n+1))
    lps = kmp_build_lps(pattern)
    res=[]; j=0
    for i in range(n):
        while j>0 and text[i]!=pattern[j]:
            j=lps[j-1]
        if text[i]==pattern[j]:
            j+=1
            if j==m:
                res.append(i-m+1)
                j=lps[j-1]
    return res

# -----------------------------------------------------------------------------
# 19. Approximate Vertex Cover (2-approx greedy)
# -----------------------------------------------------------------------------

def approx_vertex_cover(edges: List[Tuple[Any, Any]]) -> Set[Any]:
    """
    2-approximation: pick an arbitrary edge (u,v), add both u and v to cover,
    remove incident edges, and repeat.
    """
    E = set(tuple(sorted(e)) for e in edges)
    cover=set()
    while E:
        u,v = next(iter(E))
        cover.update([u,v])
        # remove incident edges
        E = set((a,b) for (a,b) in E if a not in (u,v) and b not in (u,v))
    return cover

# -----------------------------------------------------------------------------
# 20. Approximate Subset Sum (trim algorithm, epsilon in (0,1))
# -----------------------------------------------------------------------------

def approx_subset_sum(S: List[int], t: int, epsilon: float=0.2) -> int:
    """
    Return a value <= t that approximates the subset sum within (1+epsilon) factor
    of optimal. Time roughly O(n/epsilon log t).
    """
    L = [0]
    for x in S:
        L = sorted(set(L + [l + x for l in L]))
        # trim
        new=[L[0]]
        for y in L[1:]:
            if y > new[-1]*(1+epsilon):
                new.append(y)
        # remove > t
        L = [y for y in new if y <= t] or [min(new)]
    return max([y for y in L if y <= t], default=0)

# -----------------------------------------------------------------------------
# 21. Binomial Heap (insert, find-min, union, extract-min)
# -----------------------------------------------------------------------------

class BinomialNode:
    def __init__(self, key):
        self.key=key; self.parent=None; self.child=None; self.sibling=None; self.degree=0

class BinomialHeap:
    def __init__(self):
        self.head: Optional[BinomialNode] = None

    def _merge_roots(self, h: 'BinomialHeap') -> Optional[BinomialNode]:
        a, b = self.head, h.head
        dummy = BinomialNode(None); tail = dummy
        while a and b:
            if a.degree <= b.degree:
                tail.sibling = a; a = a.sibling
            else:
                tail.sibling = b; b = b.sibling
            tail = tail.sibling
        tail.sibling = a if a else b
        return dummy.sibling

    @staticmethod
    def _link(y: BinomialNode, z: BinomialNode):
        # make y child of z (assumes y.key >= z.key for min-heap)
        y.parent = z
        y.sibling = z.child
        z.child = y
        z.degree += 1

    def union(self, h: 'BinomialHeap') -> 'BinomialHeap':
        res = BinomialHeap()
        res.head = self._merge_roots(h)
        if not res.head: return res
        prev=None; curr=res.head; nxt=curr.sibling
        while nxt:
            if curr.degree != nxt.degree or (nxt.sibling and nxt.sibling.degree == curr.degree):
                prev, curr, nxt = curr, nxt, nxt.sibling
            else:
                if curr.key <= nxt.key:
                    curr.sibling = nxt.sibling
                    BinomialHeap._link(nxt, curr)
                else:
                    if prev: prev.sibling = nxt
                    else: res.head = nxt
                    BinomialHeap._link(curr, nxt)
                    curr = nxt
                nxt = curr.sibling
        self.head = res.head
        return self

    def insert(self, key):
        h = BinomialHeap(); node=BinomialNode(key); h.head=node
        self.union(h)

    def _extract_min_root(self) -> Tuple[Optional[BinomialNode], Optional[BinomialNode]]:
        if not self.head: return None, None
        prev=None; best_prev=None; best=self.head; curr=self.head
        while curr:
            if curr.key < best.key:
                best=curr; best_prev=prev
            prev=curr; curr=curr.sibling
        if best_prev:
            best_prev.sibling = best.sibling
        else:
            self.head = best.sibling
        return best, self.head

    def get_min(self) -> Optional[int]:
        curr=self.head
        if not curr: return None
        m=curr.key
        while curr:
            if curr.key < m: m=curr.key
            curr=curr.sibling
        return m

    def extract_min(self) -> Optional[int]:
        best, _ = self._extract_min_root()
        if not best: return None
        # reverse children to a heap
        rev=None
        c=best.child
        while c:
            nxt=c.sibling
            c.sibling = rev
            c.parent=None
            rev=c
            c=nxt
        h = BinomialHeap(); h.head=rev
        self.union(h)
        return best.key

# -----------------------------------------------------------------------------
# 22. Fibonacci Heap (insert, extract-min, delete via decrease-key)
# -----------------------------------------------------------------------------

class FibNode:
    def __init__(self, key):
        self.key=key; self.parent=None; self.child=None
        self.degree=0; self.mark=False
        self.left=self; self.right=self

def _fib_concat(a: Optional[FibNode], b: Optional[FibNode]) -> Optional[FibNode]:
    if not a: return b
    if not b: return a
    # splice b into a's circular list
    a.right.left = b.left
    b.left.right = a.right
    a.right = b
    b.left = a
    return a if a.key <= b.key else b

class FibonacciHeap:
    def __init__(self):
        self.min: Optional[FibNode] = None
        self.n = 0

    def insert(self, key) -> FibNode:
        node = FibNode(key)
        self.min = _fib_concat(self.min, node)
        self.n += 1
        return node

    def get_min(self):
        return self.min.key if self.min else None

    def extract_min(self):
        z = self.min
        if not z: return None
        # add children to root list
        x = z.child
        if x:
            kids=[]
            start=x
            while True:
                kids.append(x)
                x.parent=None
                x=x.right
                if x is start: break
            for k in kids:
                k.left = k.right = k
                self.min = _fib_concat(self.min, k)
        # remove z from root list
        z.left.right = z.right
        z.right.left = z.left
        self.n -= 1
        if z is z.right:
            self.min=None
        else:
            self.min=z.right
            self._consolidate()
        return z.key

    def _consolidate(self):
        A=[None]*(int(math.log2(self.n))+3 if self.n>0 else 1)
        # iterate roots
        roots=[]
        x=self.min
        if not x: return
        start=x
        while True:
            roots.append(x)
            x=x.right
            if x is start: break
        for w in roots:
            x=w
            d=x.degree
            while A[d] is not None:
                y=A[d]
                if y.key < x.key:
                    x,y = y,x
                # link y under x
                y.left.right = y.right
                y.right.left = y.left
                y.left = y.right = y
                y.parent = x
                y.right = x.child or y
                y.left = y.right.left
                y.right.left = y
                x.child = y
                x.degree += 1
                y.mark = False
                A[d]=None
                d+=1
            A[d]=x
        self.min=None
        for a in A:
            if a:
                a.left = a.right = a
                self.min = _fib_concat(self.min, a)

    def decrease_key(self, x: FibNode, k):
        assert k <= x.key, "new key is greater"
        x.key = k
        y = x.parent
        if y and x.key < y.key:
            self._cut(x, y)
            self._cascading_cut(y)
        if x.key < self.min.key:
            self.min = x

    def _cut(self, x: FibNode, y: FibNode):
        # remove x from y's child list
        if x.right is x:
            y.child=None
        else:
            x.right.left = x.left
            x.left.right = x.right
            if y.child is x: y.child = x.right
        y.degree -= 1
        x.left = x.right = x
        x.parent=None; x.mark=False
        self.min = _fib_concat(self.min, x)

    def _cascading_cut(self, y: FibNode):
        z = y.parent
        if z:
            if not y.mark:
                y.mark=True
            else:
                self._cut(y, z)
                self._cascading_cut(z)

    def delete(self, x: FibNode):
        self.decrease_key(x, -float('inf'))
        self.extract_min()

# -----------------------------------------------------------------------------
# 23. Approximate TSP (metric) via MST preorder walk (2-approx)
# -----------------------------------------------------------------------------

def approx_tsp_mst(coords: List[Tuple[float, float]]) -> Tuple[List[int], float]:
    """
    Given points in 2D with Euclidean distance (metric), return a tour (order of indices)
    and its length using MST preorder traversal (2-approximation).
    """
    n = len(coords)
    if n == 0: return [], 0.0
    # Prim's MST
    in_mst=[False]*n
    dist=[float('inf')]*n
    parent=[-1]*n
    dist[0]=0
    for _ in range(n):
        u = min((i for i in range(n) if not in_mst[i]), key=lambda i: dist[i])
        in_mst[u]=True
        for v in range(n):
            if not in_mst[v]:
                w = math.dist(coords[u], coords[v])
                if w < dist[v]:
                    dist[v]=w; parent[v]=u
    # build adjacency
    g=[[] for _ in range(n)]
    for v in range(1,n):
        u=parent[v]
        g[u].append(v); g[v].append(u)
    # preorder DFS
    order=[]
    stack=[0]; seen=set()
    while stack:
        u=stack.pop()
        if u in seen: continue
        seen.add(u); order.append(u)
        for v in reversed(g[u]):
            if v not in seen: stack.append(v)
    order.append(order[0])
    # compute length
    length = sum(math.dist(coords[order[i]], coords[order[i+1]]) for i in range(len(order)-1))
    return order, length

# Convenience: __all__ for wildcard imports in notebooks
__all__ = [
    # BST & Optimal BST
    "BST","BSTNode","optimal_bst_cost","build_optimal_bst",
    # searches
    "linear_search","binary_search",
    # quicksorts
    "quicksort","quicksort_pivot",
    # heaps & PQ
    "build_min_heap","build_max_heap","extract_min","insert_min",
    "kth_min_in_minheap","kth_max_in_maxheap","heap_sort","PriorityQueue",
    "analyze_marks_with_heap","delete_kth_index_minheap","delete_kth_index_maxheap",
    # insertion sort
    "insertion_sort",
    # balanced trees
    "AVLTree","RBTree","BTree",
    # string matching
    "naive_search","rabin_karp","kmp","kmp_build_lps",
    # approximations
    "approx_vertex_cover","approx_subset_sum",
    # binomial & fibonacci heaps
    "BinomialHeap","FibonacciHeap",
    # TSP approx
    "approx_tsp_mst",
]
