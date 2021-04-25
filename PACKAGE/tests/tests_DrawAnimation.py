import UnitTest.class_DrawAnimation as cd
import unittest

class TestClass_DrawAnimation(unittest.TestCase):

    def testUpdate_c(self):
        r = cd.update_c(fouriercoef, 25/ len(np.linspace(0,tau,300)) * tau) #Exemple avec i=25, space = np.linspace(0,tau,300)
        self.assertIs(type(r), ndarray)

    def testSort_velocity(self):
        idx = cd.sort_velocity(50) #Exemple avec order=50
        self.assertIs(type(idx), list)

if __name__ == "__main__":
    unittest.main()        
