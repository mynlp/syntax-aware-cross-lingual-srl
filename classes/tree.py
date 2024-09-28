class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """

    def __init__(self, idx):
        self.idx = idx
        self.parent = None
        self.deprel = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child, deprel):
        child.parent = self
        child.deprel = deprel
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size', False):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def height(self):
        if getattr(self, '_height', False):
            return self._height
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_height = self.children[i].height()
                if child_height > count:
                    count = child_height
            count += 1
        self._height = count
        return self._height

    def depth(self):
        if getattr(self, '_depth', False):
            return self._depth
        count = 0
        if self.parent:
            count += self.parent.depth()
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x
