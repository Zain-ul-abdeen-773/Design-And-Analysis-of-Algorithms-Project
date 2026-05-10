import numpy as np
from typing import List, Dict, Tuple
import warnings
from collections import defaultdict

try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        CUDA_AVAILABLE = True
        DEFAULT_DEVICE = 'cuda'
    else:
        CUDA_AVAILABLE = False
        DEFAULT_DEVICE = 'cpu'
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    DEFAULT_DEVICE = 'cpu'
    warnings.warn("PyTorch not available. Tensor Eclat will use NumPy fallback.")

class TensorEclat:
    """
    State-of-the-Art GPU-Accelerated Frequent Itemset Mining using Support-Ascending Heuristic.
    """
    def __init__(self, min_support: float, device: str = None):
        self.min_support = min_support
        
        if device is None:
            self.device = DEFAULT_DEVICE
        else:
            self.device = device
            
        self.use_torch = TORCH_AVAILABLE
        
        if self.use_torch:
            self.torch_device = torch.device(self.device)
            
        self.num_transactions = 0
        self.frequent_itemsets = {}
        self.candidate_count = 0

    def fit(self, transactions: List[List[int]]) -> Dict[int, Dict[Tuple[int, ...], int]]:
        self.num_transactions = len(transactions)
        min_sup_count = int(np.ceil(self.min_support * self.num_transactions))
        self.frequent_itemsets = {}
        self.candidate_count = 0

        # 1. Vertical Data Format Conversion
        item_to_tid = defaultdict(list)
        for tid, t in enumerate(transactions):
            for item in t:
                item_to_tid[item].append(tid)

        frequent_1 = {item: ptids for item, ptids in item_to_tid.items() if len(ptids) >= min_sup_count}
        
        if not frequent_1:
            return self.frequent_itemsets

        self.frequent_itemsets[1] = {(item,): len(ptids) for item, ptids in frequent_1.items()}
        
        # 2. Support-Ascending Heuristic
        # Sorting items by their support in ascending order dramatically prunes the search tree.
        items_sorted_by_support = sorted(frequent_1.keys(), key=lambda x: len(frequent_1[x]))

        # 3. Create boolean tensors (or numpy arrays)
        if self.use_torch:
            suffix = []
            for item in items_sorted_by_support:
                tids = frequent_1[item]
                t_tensor = torch.zeros(self.num_transactions, dtype=torch.bool, device=self.torch_device)
                t_tensor[tids] = True
                suffix.append((item, t_tensor))
                
            # Use torch.no_grad to prevent memory leaks during recursive calls
            with torch.no_grad():
                for i in range(len(suffix)):
                    item, tidset = suffix[i]
                    self._eclat_dfs_torch((item,), tidset, suffix[i + 1:], min_sup_count)
        else:
            # Numpy Fallback
            suffix = []
            for item in items_sorted_by_support:
                tids = frequent_1[item]
                t_arr = np.zeros(self.num_transactions, dtype=bool)
                t_arr[tids] = True
                suffix.append((item, t_arr))
                
            for i in range(len(suffix)):
                item, tidset = suffix[i]
                self._eclat_dfs_numpy((item,), tidset, suffix[i + 1:], min_sup_count)

        return self.frequent_itemsets

    def _eclat_dfs_torch(self, prefix: Tuple[int, ...], prefix_tidset: 'torch.Tensor',
                         suffix_items: List[Tuple[int, 'torch.Tensor']],
                         min_sup_count: int):
        new_frequent_pairs = []

        for item, item_tidset in suffix_items:
            self.candidate_count += 1

            new_tidset = torch.bitwise_and(prefix_tidset, item_tidset)
            support = int(new_tidset.sum().item())

            if support >= min_sup_count:
                new_itemset = tuple(sorted(prefix + (item,)))
                k = len(new_itemset)

                if k not in self.frequent_itemsets:
                    self.frequent_itemsets[k] = {}
                self.frequent_itemsets[k][new_itemset] = support
                new_frequent_pairs.append((item, new_tidset))

        for i in range(len(new_frequent_pairs)):
            item, tidset = new_frequent_pairs[i]
            remaining = new_frequent_pairs[i + 1:]
            if remaining:
                self._eclat_dfs_torch(prefix + (item,), tidset, remaining, min_sup_count)
                
    def _eclat_dfs_numpy(self, prefix: Tuple[int, ...], prefix_tidset: np.ndarray,
                         suffix_items: List[Tuple[int, np.ndarray]],
                         min_sup_count: int):
        new_frequent_pairs = []

        for item, item_tidset in suffix_items:
            self.candidate_count += 1

            new_tidset = np.logical_and(prefix_tidset, item_tidset)
            support = int(new_tidset.sum())

            if support >= min_sup_count:
                new_itemset = tuple(sorted(prefix + (item,)))
                k = len(new_itemset)

                if k not in self.frequent_itemsets:
                    self.frequent_itemsets[k] = {}
                self.frequent_itemsets[k][new_itemset] = support
                new_frequent_pairs.append((item, new_tidset))

        for i in range(len(new_frequent_pairs)):
            item, tidset = new_frequent_pairs[i]
            remaining = new_frequent_pairs[i + 1:]
            if remaining:
                self._eclat_dfs_numpy(prefix + (item,), tidset, remaining, min_sup_count)
