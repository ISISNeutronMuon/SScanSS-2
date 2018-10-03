from ..math.vector import Vector4
from ..math.algorithm import clamp


class Colour:
    def __init__(self, red, green, blue, alpha=1.0):
        self.__colour = Vector4()
        self.r = red
        self.g = green
        self.b = blue
        self.a = alpha

    @property
    def r(self):
        return self.__colour.x

    @r.setter
    def r(self, value):
        self.__colour.x = clamp(value)

    @property
    def g(self):
        return self.__colour.y

    @g.setter
    def g(self, value):
        self.__colour.y = clamp(value)

    @property
    def b(self):
        return self.__colour.z

    @b.setter
    def b(self, value):
        self.__colour.z = clamp(value)

    @property
    def a(self):
        return self.__colour.w

    @a.setter
    def a(self, value):
        self.__colour.w = clamp(value)

    def invert(self):
        return Colour(1-self.r, 1-self.g, 1-self.b, self.a)

    @property
    def rgba(self):
        return (self.__colour[:] * 255).astype(int)

    @property
    def rgbaf(self):
        return self.__colour[:]

    @staticmethod
    def normalize(r=0, g=0, b=0, a=255):
        c = Colour(r/255, g/255, b/255, a/255)

        return c
    
    @staticmethod
    def white():
        return Colour(1.0, 1.0, 1.0)

    @staticmethod
    def black():
        return Colour(0.0, 0.0, 0.0)

    def __getitem__(self, index):
        return self.__colour[index]

    def __str__(self):
            return 'rgba({}, {}, {}, {})'.format(self.r, self.g, self.b, self.a)

    def __repr__(self):
        return 'Colour({}, {}, {}, {})'.format(self.r, self.g, self.b, self.a)
