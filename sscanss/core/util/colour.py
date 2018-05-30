from pyrr import Vector4


class Colour:
    def __init__(self, red, green, blue, alpha=1.0):
        self.__colour = Vector4([red, green, blue, alpha])

    @property
    def r(self):
        return self.__colour.x

    @r.setter
    def r(self, val):
        self.__colour.x = self.__normalize(val)

    @property
    def g(self):
        return self.__colour.y

    @g.setter
    def g(self, val):
        self.__colour.y = self.__normalize(val)

    @property
    def b(self):
        return self.__colour.z

    @b.setter
    def b(self, val):
        self.__colour.z = self.__normalize(val)

    @property
    def a(self):
        return self.__colour.w

    @a.setter
    def a(self, val):
        self.__colour.w = self.__normalize(val)

    def invert(self):
        return Colour(1-self.r, 1-self.g, 1-self.b, self.a)

    def __normalize(self, val):
        if 0.0 <= val <= 1.0:
            return val
        elif val < 0.0:
            return 0.0
        else:
            return 1.0

    def rgba(self):
        return self.__colour * 255

    def rgbaf(self):
        return self.__colour

    @staticmethod
    def normalize(r=0, g=0, b=0, a=255):
        c = Colour(r, g, b, a)
        c.__colour /= 255

        return c
    
    @staticmethod
    def white():
        return Colour(1.0, 1.0, 1.0)

    @staticmethod
    def black():
        return Colour(0.0, 0.0, 0.0)

    def __getitem__(self, index):
        if index > 3:
            raise IndexError('not a valid index')
        
        return self.__colour[index]


    def __str__(self):
            return 'RGBA({}, {}, {}, {})'.format(self.r, self.g, self.b, self.a)

    def __repr__(self):
        return 'RGBA({}, {}, {}, {})'.format(self.r, self.g, self.b, self.a)
