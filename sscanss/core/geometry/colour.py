"""
Class representing colour
"""
from ..math.vector import Vector4
from ..math.misc import clamp


class Colour:
    """Creates Colour object that represents a normalized [0, 1] RGBA colour.

    :param red: Red channel value between 0 and 1
    :type red: float
    :param green: Green channel value between 0 and 1
    :type green: float
    :param blue: Blue channel value between 0 and 1
    :type blue: float
    :param alpha: Alpha channel value between 0 and 1.
    :type alpha: float
    """
    def __init__(self, red, green, blue, alpha=1.0):
        self.__colour = Vector4()
        self.r = red
        self.g = green
        self.b = blue
        self.a = alpha

    @property
    def r(self):
        """Gets and sets value of red channel

        :return: red value
        :rtype: float
        """
        return self.__colour.x

    @r.setter
    def r(self, value):
        self.__colour.x = clamp(value)

    @property
    def g(self):
        """Gets and sets value of green channel

        :return: green value
        :rtype: float
        """
        return self.__colour.y

    @g.setter
    def g(self, value):
        self.__colour.y = clamp(value)

    @property
    def b(self):
        """Gets and sets value of blue channel

        :return: blue value
        :rtype: float
        """
        return self.__colour.z

    @b.setter
    def b(self, value):
        self.__colour.z = clamp(value)

    @property
    def a(self):
        """Gets and sets value of alpha channel

        :return: alpha value
        :rtype: float
        """
        return self.__colour.w

    @a.setter
    def a(self, value):
        self.__colour.w = clamp(value)

    def invert(self):
        """Inverts the RGB channels i.e (1-r, 1-g, 1-b, a) of colour

        :return: inverse of colour
        :rtype: Colour
        """
        return Colour(1-self.r, 1-self.g, 1-self.b, self.a)

    @property
    def rgba(self):
        """Gets un-normalized colour values

        :return: un-normalized RGBA colour [0-255]
        :rtype: numpy.ndarray
        """
        return (self.__colour[:] * 255).astype(int)

    @property
    def rgbaf(self):
        """Gets normalized colour values

        :return: normalized RGBA colour [0-1]
        :rtype: numpy.ndarray
        """
        return self.__colour[:]

    @staticmethod
    def normalize(red=0, green=0, blue=0, alpha=255):
        """Create Colour object by converting to normalized RGBA from
        un-normalized values

        :param red: Red channel value between 0 and 255
        :type red: int
        :param green: Green channel value between 0 and 255
        :type green: int
        :param blue: Blue channel value between 0 and 255
        :type blue: int
        :param alpha: Alpha channel value between 0 and 255.
        :type alpha: int
        :return: normalized RGBA colour
        :rtype: Colour
        """
        return Colour(red/255, green/255, blue/255, alpha/255)
    
    @staticmethod
    def white():
        """Creates white colour

        :return: white colour
        :rtype: Colour
        """
        return Colour(1.0, 1.0, 1.0)

    @staticmethod
    def black():
        """Creates black colour

        :return: black colour
        :rtype: Colour
        """
        return Colour(0.0, 0.0, 0.0)

    def __getitem__(self, index):
        return self.__colour[index]

    def __str__(self):
        return 'rgba({}, {}, {}, {})'.format(self.r, self.g, self.b, self.a)

    def __repr__(self):
        return 'Colour({}, {}, {}, {})'.format(self.r, self.g, self.b, self.a)
