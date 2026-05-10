import itertools
from collections import defaultdict
from typing import List, Tuple, Set, Dict

class AprioriBaseline:
    """
    Optimized implementation of the Apriori Algorithm using Vertical Data Format (Tidsets).
    
    This replaces the naive database scanning with bitwise/set intersections,
    achieving a massive speedup on sparse to moderately dense datasets.
    """
    def __init__(self, min_support: float):
        self.min_support = min_support
        self.num_transactions = 0
        self.min_sup_count = 0
        self.frequent_itemsets = {}  # { k: { itemset_tuple: support } }
        self.candidate_count = 0
        self.tidsets = {}  # { itemset_frozenset: set_of_tids }

    def _build_vertical_database(self, transactions: List[frozenset]) -> Dict[int, Set[int]]:
        """Convert horizontal database to vertical database (item -> set of tids)."""
        vertical_db = defaultdict(set)
        for tid, transaction in enumerate(transactions):
            for item in transaction:
                vertical_db[item].add(tid)
        return dict(vertical_db)

    def _compute_tidset_intersection(self, itemset1: frozenset, itemset2: frozenset) -> Set[int]:
        """Compute the intersection of tidsets for two itemsets."""
        tidset1 = self.tidsets.get(itemset1, set())
        tidset2 = self.tidsets.get(itemset2, set())
        return tidset1 & tidset2

    def _generate_candidates_with_tidsets(self, prev_frequent: Dict[frozenset, int], k: int) -> Dict[frozenset, int]:
        """Generate candidates and compute support simultaneously using tidset intersection."""
        prefix_groups = defaultdict(list)
        for itemset in prev_frequent.keys():
            sorted_items = tuple(sorted(itemset))
            prefix = sorted_items[:-1]
            prefix_groups[prefix].append((sorted_items[-1], itemset))
        
        frequent = {}
        
        for prefix, items_list in prefix_groups.items():
            items_list.sort(key=lambda x: x[0])
            
            for i in range(len(items_list)):
                for j in range(i + 1, len(items_list)):
                    _, itemset1 = items_list[i]
                    _, itemset2 = items_list[j]
                    
                    new_candidate = itemset1 | itemset2
                    self.candidate_count += 1
                    
                    # Apriori Pruning Step
                    is_valid = True
                    for subset in itertools.combinations(new_candidate, k - 1):
                        if frozenset(subset) not in prev_frequent:
                            is_valid = False
                            break
                    
                    if not is_valid:
                        continue
                    
                    # Compute support via set intersection
                    new_tidset = self._compute_tidset_intersection(itemset1, itemset2)
                    support = len(new_tidset)
                    
                    if support >= self.min_sup_count:
                        self.tidsets[new_candidate] = new_tidset
                        frequent[new_candidate] = support
        
        return frequent

    def fit(self, transactions: List[List[int]]) -> Dict[int, Dict[Tuple[int, ...], int]]:
        transactions_fs = [frozenset(t) for t in transactions]
        self.num_transactions = len(transactions_fs)
        self.min_sup_count = self.min_support * self.num_transactions
        
        self.frequent_itemsets = {}
        self.tidsets = {}
        self.candidate_count = 0
        
        # Build vertical database
        vertical_db = self._build_vertical_database(transactions_fs)
        
        # Level 1
        current_frequent = {}
        for item, tidset in vertical_db.items():
            if len(tidset) >= self.min_sup_count:
                itemset = frozenset([item])
                current_frequent[itemset] = len(tidset)
                self.tidsets[itemset] = tidset
                
        k = 1
        while current_frequent:
            # Store in original format: Tuple[int, ...]
            self.frequent_itemsets[k] = {tuple(sorted(fs)): count for fs, count in current_frequent.items()}
            k += 1
            
            # Generate and count candidates for next level
            current_frequent = self._generate_candidates_with_tidsets(current_frequent, k)
            
            # Free up memory of older tidsets
            if k > 2:
                for old_itemset in list(self.frequent_itemsets.get(k-2, {}).keys()):
                    self.tidsets.pop(frozenset(old_itemset), None)
                    
        return self.frequent_itemsets

    def generate_rules(self, min_confidence: float = 0.7) -> List[Dict]:
        """
        Generate association rules from discovered frequent itemsets.
        """
        support_lookup = {}
        for k, itemsets in self.frequent_itemsets.items():
            for itemset, count in itemsets.items():
                support_lookup[itemset] = count

        rules = []
        for k, itemsets in self.frequent_itemsets.items():
            if k < 2:
                continue
            for itemset, sup_count in itemsets.items():
                for i in range(1, len(itemset)):
                    for antecedent in itertools.combinations(itemset, i):
                        consequent = tuple(sorted(set(itemset) - set(antecedent)))
                        ant_support = support_lookup.get(antecedent, 0)
                        if ant_support > 0:
                            confidence = sup_count / ant_support
                            if confidence >= min_confidence:
                                rules.append({
                                    'antecedent': antecedent,
                                    'consequent': consequent,
                                    'support': sup_count / self.num_transactions,
                                    'confidence': confidence,
                                })
        return rules