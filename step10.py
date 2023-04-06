import unittest
import numpy as np
from utils import Variable, square, numerical_diff

class SquareTest(unittest.TestCase):
    """
    unittest.TestCase를 상속 받아서 만들고
    
    메서드 이름은 항상 test~ 로시작해야함

    Args:
        unittest (_type_): _description_
    """
    
    def test_foward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
        
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(1)
        self.assertEqual(x.grad,expected)
        
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
        
        
unittest.main() #를 입력하면 'step10.py'만 입력해도 test할 수 있음