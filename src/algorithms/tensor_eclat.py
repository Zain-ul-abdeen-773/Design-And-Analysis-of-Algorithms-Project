import torch
import numpy as np
from typing import List, Tuple, Dict

class TensorEclat:
    """
    State-of-the-Art GPU-Accelerated Frequent Itemset Mining (2022+ Approach)
    
    This implements a highly optimized vertical data format using Tensor operations 
    on PyTorch/CuPy, parallelizing support counting massively [cite: 41, 55, 59, 60].
    Memory-efficient bitsets (boolean tensors) eliminate repeat db scans[cite: 57].

    The algorithm uses a Depth-First Search (DFS) strategy through equivalence
    classes, performing support counting via GPU-parallel bitwise AND operations.
    """
    def __init__(self, min_support: float, device: str = 'cuda'):
        self.min_support = min_support
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_transactions = 0
        self.frequent_itemsets = {}
        self.candidate_count = 0

    def fit(self, transactions: List[List[int]]) -> Dict[int, Dict[Tuple[int], int]]:
        self.num_transactions = len(transactions)
        min_sup_count = self.min_support * self.num_transactions

        # 1. Optimisation 1: Vertical Data Format Conversion
        item_to_tid = {}
        for tid, t in enumerate(transactions):
            for item in t:
                if item not in item_to_tid:
                    item_to_tid[item] = []
                item_to_tid[item].append(tid)

        # Filter length 1 itemsets
        frequent_1 = {item: ptids for item, ptids in item_to_tid.items() if len(ptids) >= min_sup_count}
        self.frequent_itemsets[1] = {(item,): len(ptids) for item, ptids in frequent_1.items()}
        
        # 2. Optimisation 2: GPU Tensors (Boolean vectors mapped to GPU)
        self.tensor_db = {}
        for item, tids in frequent_1.items():
            t_tensor = torch.zeros(self.num_transactions, dtype=torch.bool, device=self.device)
            t_tensor[tids] = True
            self.tensor_db[item] = t_tensor

        # DFS mining through equivalence classes
        items = sorted(frequent_1.keys())
        suffix = [(item, self.tensor_db[item]) for item in items]

        for i in range(len(suffix)):
            item, tidset = suffix[i]
            remaining = suffix[i + 1:]
            self._eclat_dfs((item,), tidset, remaining, min_sup_count)

        return self.frequent_itemsets

    def _eclat_dfs(self, prefix: Tuple[int], prefix_tidset: torch.Tensor,
                   suffix_items: List[Tuple[int, torch.Tensor]],
                   min_sup_count: float):
        """
        Recursive Depth-First Search through equivalence classes.

        At each level, intersect the prefix tidset with each suffix item's tidset
        using GPU-parallel bitwise AND. Frequent extensions form the new equivalence
        class for deeper recursion.
        """
        new_frequent_pairs = []

        for item, item_tidset in suffix_items:
            self.candidate_count += 1

            # GPU Parallel Intersection (Bitwise AND across CUDA cores)
            new_tidset = torch.bitwise_and(prefix_tidset, item_tidset)
            support = int(new_tidset.sum().item())

            if support >= min_sup_count:
                new_itemset = tuple(sorted(prefix + (item,)))
                k = len(new_itemset)

                if k not in self.frequent_itemsets:
                    self.frequent_itemsets[k] = {}
                self.frequent_itemsets[k][new_itemset] = support
                new_frequent_pairs.append((item, new_tidset))

        # Recurse deeper into equivalence classes
        for i in range(len(new_frequent_pairs)):
            item, tidset = new_frequent_pairs[i]
            remaining = new_frequent_pairs[i + 1:]
            if remaining:
                self._eclat_dfs(prefix + (item,), tidset, remaining, min_sup_count)
