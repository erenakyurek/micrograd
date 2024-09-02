import math

class Value:

# CONSTRUCTOR
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._backward = lambda: None
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op

# TOSTRING
    def __repr__(self):
        return f"Value(data={self.data})"

# __ADD__
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

# __RADD__
    def __radd__(self, other):
        return self + other

# __NEG__
    def __neg__(self):
        return self * -1

# __SUB__
    def __sub__(self, other):
        return self + (-other)

# __RSUB__
    def __rsub__(self, other):
        return (-self) + other

# __MUL__
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

# RMUL
    def __rmul__(self, other):
        return self * other

# EXP
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

# TRUE_DIV
    def __truediv__(self, other):
        return self * other**-1

# POW
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

# BACKWARD
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

# TANH
    def tanh(self):
        x = self.data
        t = ((math.exp(2*x) - 1)/(math.exp(2*x) + 1))
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
