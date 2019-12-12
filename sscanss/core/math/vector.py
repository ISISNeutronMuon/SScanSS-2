"""
Classes for Vector operations
"""

import numpy as np


class Vector:
    """Creates Vector object with specified size. if values are not
    provided the matrix will be initialized with zeros.

    :param size: size of vector
    :type size: int
    :param values: value to populate vector
    :type values: array-like
    :param dtype: data type of vector.
    :type dtype: numpy.dtype
    :raises: ValueError
    """
    __array_priority__ = 1

    def __init__(self, size, values=None, dtype=None):
        if size < 1:
            raise ValueError("size must not be less than 1")
        if values is not None:
            if len(values) != size:
                raise ValueError("Values does not match specified size!")
            data = np.array(values[:size], dtype)
        else:
            data = np.zeros(size, dtype)
        
        super().__setattr__("size", size)
        super().__setattr__("_data", data)
        super().__setattr__("_keys", {})

    def __array__(self, _dtype=None):
        return self._data

    def __getattr__(self, attr):
        if attr == "__setstate__":
            # fix for recursion problem during copy
            raise AttributeError(attr)
        if attr in self._keys:
            index = self._keys[attr]
            return self._data[index]
        else:
            raise AttributeError("'Vector' object has no attribute '{}'".format(attr))

    def __setattr__(self, attr, value):
        if attr in self._keys:
            index = self._keys[attr]
            self._data[index] = value
            return
        elif hasattr(self, attr):
            super().__setattr__(attr, value)
            return
        else:
            raise AttributeError("'Vector' object has no attribute '{}'".format(attr))

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value
    
    @staticmethod
    def create(size, data=None):
        if size == 2:
            return Vector2(data)
        elif size == 3:
            return Vector3(data)
        elif size == 4:
            return Vector4(data)
        else:
            return Vector(size, data)

    @property
    def length(self):
        return np.linalg.norm(self._data)

    def normalize(self):
        self._data = self._data / self.length

    @property
    def normalized(self):
        data = self._data / self.length
        return self.create(self.size, data)

    def __helper(self, left, right, func):
        result = func(left, right)

        if isinstance(result, np.ndarray) and result.size == self.size and result.ndim == 1:
            return self.create(self.size, result)

        return result

    def __add__(self, other):
        func = np.add
        if isinstance(other, Vector):
            if len(other) != self.size:
                raise ValueError("cannot add vectors of different sizes")
            return self.__helper(self._data, other[:], func)
        else:
            return self.__helper(self._data, other, func)

    def __sub__(self, other):
        func = np.subtract
        if isinstance(other, Vector):
            if len(other) != self.size:
                raise ValueError("cannot subtract vectors of different sizes")
            return self.__helper(self._data, other[:], func)
        else:
            return self.__helper(self._data, other, func)

    def __mul__(self, other):
        func = np.multiply
        if isinstance(other, Vector):
            if len(other) != self.size:
                raise ValueError("cannot multiply vectors of different sizes")
            return self.__helper(self._data, other[:], func)
        else:
            return self.__helper(self._data, other, func)

    def __truediv__(self, other):
        func = np.true_divide
        if isinstance(other, Vector):
            if len(other) != self.size:
                raise ValueError("cannot divide vectors of different sizes")
            return self.__helper(self._data, other[:], func)
        else:
            return self.__helper(self._data, other, func)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__helper(other, self._data, np.subtract)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__helper(other, self._data, np.true_divide)

    def __iadd__(self, other):
        temp = self.__add__(other)
        self._data = temp._data
        return self

    def __matmul__(self, other):
        func = np.matmul
        if isinstance(other, (Vector, np.ndarray)):
            if len(other) != self.size:
                raise ValueError("cannot matmul vectors of different sizes")
            return self.__helper(self._data, other[:], func)
        else:
            return self.__helper(self._data, other, func)

    def __rmatmul__(self, other):
        return self.__helper(other, self._data, np.matmul)

    def __isub__(self, other):
        temp = self.__sub__(other)
        self._data = temp._data
        return self

    def __imul__(self, other):
        temp = self.__mul__(other)
        self._data = temp._data
        return self

    def dot(self, other):
        return np.dot(self._data, other[:])

    def cross(self, other):
        data = np.cross(self._data, other[:])
        if data.size == 1:
            return self.create(3, [0, 0, data])
        else:    
            return self.create(data.size, data)

    def __or__(self, other):
        return self.dot(other)
    
    def __xor__(self, other):
        return self.cross(other)

    def __neg__(self):
        return self.create(self._data.size, -1 * self._data)

    def __len__(self):
        return self.size

    def __str__(self):
        return str(self._data)


class Vector2(Vector):
    """Creates a 2D Vector.

    :param values: value to populate vector
    :type values: array-like
    :param dtype: data type of vector.
    :type dtype: numpy.dtype
    :raises: ValueError
    """
    def __init__(self, values=None, dtype=None):
        super().__init__(2, values, dtype)
        self._keys = {'x': 0, 'y': 1, 'xy': slice(None)}


class Vector3(Vector):
    """Creates a 3D Vector.

    :param values: value to populate vector
    :type values: array-like
    :param dtype: data type of vector.
    :type dtype: numpy.dtype
    :raises: ValueError
    """
    def __init__(self, values=None, dtype=None):
        super().__init__(3, values, dtype)
        self._keys = {'x': 0, 'y': 1, 'z': 2,
                      'xy': slice(2), 'xyz': slice(None)}


class Vector4(Vector):
    """Creates a 4D Vector.

    :param values: value to populate vector
    :type values: array-like
    :param dtype: data type of vector. Similar to Numpy
    :type dtype: numpy.dtype
    :raises: ValueError
    """
    def __init__(self, values=None, dtype=None):
        super().__init__(4, values, dtype)
        self._keys = {'x': 0, 'y': 1, 'z': 2, 'w': 3,
                      'xy': slice(2), 'xyz': slice(3), 'xyzw': slice(None)}
