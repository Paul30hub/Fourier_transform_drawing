import UnitTest.Fourier as f
import unittest

class TestClass_Fourier(unittest.TestCase):
    
    def testCoef_list(self):
        c = f.coef_list(np.linspace(0,tau,len(contour_path.vertices[:,0])), contour_path.vertices[:,0], contour_path.vertices[:,1], 50) # Exemple avec order = 50
        self.assertIs(type(c), ndarray)
    
    def testdft(self):
        d = f.DFT(2, [-0.466329862, 0.546912553], 50) #Exemple avec t=2, coef_list=([-0.466329862, 0.546912553] et order=50
        self.assertIs(type(d), list)

        
if __name__ == "__main__":
    unittest.main()        
