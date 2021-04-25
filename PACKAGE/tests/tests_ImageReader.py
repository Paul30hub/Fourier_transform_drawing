    import UnitTest.ImageReader as ir
import unittest

class TestClass_ImageReader(unittest.TestCase):

    def testGet_tour(self):
        r = ir.get_tour(np.linspace(0,tau,len(contour_path.vertices[:,0])), contour_path.vertices[:,0], contour_path.vertices[:,1])
        self.assertIs(type(r[0]), tuple)
        self.assertIs(type(r[1]), int)
        self.assertIs(type(r[2]), int)

if __name__ == "__main__":
    unittest.main()        
