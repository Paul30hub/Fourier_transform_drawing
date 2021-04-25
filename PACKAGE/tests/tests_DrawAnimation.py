import UnitTest.class_DrawAnimation as cd
import unittest

class TestClass_DrawAnimation(unittest.TestCase):

    def testUpdate_c(self):
        r = cd.update_c(coef, 25/ len(space) * tau) #Exemple avec i=25
        self.assertIs(type(r), ndarray)

    def testSort_velocity(self):
        idx = cd.sort_velocity(50) #Exemple avec order=50
        self.assertIs(type(idx), list)

if __name__ = "__main__":
    unittest.main()        
