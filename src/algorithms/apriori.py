import itertools
from collections import defaultdict
from typing import List, Tuple, Set, Dict

class AprioriBaseline:
    """
    Baseline implementation of the Apriori Algorithm using Breadth-First Search.
    
    Ref[cite: 21]: Classical level-wise candidate generation and support counting.
    """
    def __init__(self, min_support: float):
        self.min_support = min_support
        self.num_transactions = 0
        self.frequent_itemsets = {}  # { k: { itemset: support } }
        self.candidate_count = 0

    def get_frequent_1_itemsets(self, transactions: List[Set[int]]) -> Dict[Tuple[int], int]:
        counts = defaultdict(int)
        for t in transactions:
            for item in t:
                counts[(item,)] += 1
        return {item: count for item, count in counts.items() 
                if count >= self.min_support * self.num_transactions}

    def generate_candidates(self, prev_frequent: Dict[Tuple[int], int], k: int) -> Set[Tuple[int]]:
        """Join step and pruning step."""
        items = list(prev_frequent.keys())
        candidates = set()
        
        # Simple Join
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][:-1] == items[j][:-1]:
                    new_itemset = tuple(sorted(set(items[i] + items[j])))
                    if len(new_itemset) == k:
                        # Pruning: All subsets must be frequent
                        is_valid = True
                        for subset in itertools.combinations(new_itemset, k - 1):
                            if subset not in prev_frequent:
                                is_valid = False
                                break
                        if is_valid:
                            candidates.add(new_itemset)
                            
        self.candidate_count += len(candidates)
        return candidates

    def fit(self, transactions: List[List[int]]):
        self.num_transactions = len(transactions)
        transactions_set = [set(t) for t in transactions]
        
        # Level 1
        current_frequent = self.get_frequent_1_itemsets(transactions_set)
        k = 1
        
        while current_frequent:
            self.frequent_itemsets[k] = current_frequent
            k += 1
            
            # Generate C_k
            candidates = self.generate_candidates(current_frequent, k)
            if not candidates:
                break
                
            # Scan DB
            new_frequent = {}
            for t in transactions_set:
                for c in candidates:
                    if set(c).issubset(t):
                        new_frequent[c] = new_frequent.get(c, 0) + 1
                        
            # Filter
            min_sup_count = self.min_support * self.num_transactions
            current_frequent = {c: sup for c, sup in new_frequent.items() if sup >= min_sup_count}

        return self.frequent_itemsets

    def generate_rules(self, min_confidence: float = 0.7) -> List[Dict]:
        """
        Generate association rules from discovered frequent itemsets.

        For each frequent itemset of size >= 2, generate all possible rules
        X -> Y where X union Y = itemset, and check confidence threshold.
        confidence(X -> Y) = support(X union Y) / support(X)
        """
        # Build flat lookup: itemset -> support_count
        support_lookup = {}
        for k, itemsets in self.frequent_itemsets.items():
            for itemset, count in itemsets.items():
                support_lookup[itemset] = count

        rules = []
        for k, itemsets in self.frequent_itemsets.items():
            if k < 2:
                continue
            for itemset, sup_count in itemsets.items():
                # Generate all non-empty proper subsets as antecedents
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