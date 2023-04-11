from .baseline import Function, Variable

class Square(Function):
    def forward(self, x):
        y = x**2
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = 2*x*gy
        return gy
    
    
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) *gy
        return gx
    
class Add(Function):
    def forward(sel, x0, x1):
        y = x0 + x1
        
        return y

def add(x0, x1):
    return Add()(x0, x1)

def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)



def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    
    return (y1.data - y0.data) / (2 * eps)