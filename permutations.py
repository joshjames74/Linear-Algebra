from itertools import permutations


class Permutation():
    '''Representation of a permutation'''

    def __init__(self, l: list, index=0):
        if type(l) != list:
            raise Exception('Not a list')
        is_integer = [type(x) == int for x in l]
        if not all(is_integer):
            raise Exception('Contains non-integers')
        # if sorted(l) != list(range(index, max(l)+1)):
        #     raise Exception('Invalid permutation')
        self.l = l
    
    def __str__(self) -> str:
        '''Return string representation of permutation'''
        return str(tuple(self.l))

    def apply(self, x: int):
        '''Apply permutation to an integer
        '''
        if type(x) != int:
            raise Exception('Not an integer')
        return self.l[x]
    
    def decomposition(self) -> list:
        '''Return representation of permutation
        as product of orbits
        '''
        decomp = []
        included_elements = []
        # Repeat until all elements are included
        while sorted(included_elements) != sorted(self.l):
            cycle = []
            # Begin with element not yet included
            excluded_elements = list(set(self.l) - set(included_elements))
            start = excluded_elements[0]
            # Add elements to cycle and included elems
            cycle.append(start)
            # Get a cycle
            current = self.apply(start)
            while current != start:
                cycle.append(current)
                current = self.apply(current)
            # Add to included elements and decomp
            included_elements += cycle
            decomp.append(cycle)
        return decomp
    
    def sign(self) -> int:
        '''Return sign of permutation'''
        decomp = self.decomposition()
        power_sum = sum([len(cycle)-1 for cycle in decomp])
        sign = (-1) ** power_sum
        return sign
            

class PermutationGroup:
    '''Representation of all permutations of a list'''

    def __init__(self, s: list):
        # Check that s is a list
        if type(s) != list:
            raise Exception('Not a list')
        # Check that that all values of s are integers
        is_integer = [type(x) == int for x in s]
        if not all(is_integer):
            raise Exception('Contains non-integers')
        self.s = s
    
    def __str__(self) -> str:
        '''Return string representation of permutation'''
        perms = []
        for group in self.group():
            perms.append(tuple(group.l))
        return str(perms)

    def group_list(self) -> list:
        '''Return list of permutations as lists'''
        group_list = list(permutations(self.s))
        return group_list
    
    def group(self) -> list:
        '''Return list of permutation as
        permutation objects
        '''
        # Need to sort out typing on
        # init of permutations to allow
        # for tuples
        group_list = self.group_list()
        return [Permutation(list(group)) for group in group_list]
