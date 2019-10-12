class Pipeline(object):
    def __init__(self):
        self.source = None

    def __iter__(self):
        return self.generator()

    def generator(self):
        while self.has_next():
            data = next(self.source) if self.source else {}
            if self.filter(data):
                yield self.map(data)

    def __or__(self, other):
        other.source = self.generator()
        return other

    def filter(self, data):
        return True

    def map(self, data):
        return data

    def has_next(self):
        return True
