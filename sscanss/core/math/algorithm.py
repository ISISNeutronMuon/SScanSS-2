def clamp(value, min_value=0.0, max_value=1.0):
    """ returns original value if it is between a minimum and maximum value
    returns minimum value if original value is less than minimum value
    returns maximum value if original value is greater than maximum value

    :param value: number to clamp
    :type value: float
    :param min_value: maximum value
    :type min_value: float
    :param max_value: minimum value
    :type max_value: float
    :return: number clamped between the specified range
    :rtype: float
    """
    return max(min(value, max_value), min_value)


def map_range(old_min, old_max, new_min, new_max, value):
    """ takes two ranges and a real number, and returns the mapping of the
    real number from the first to the second range.

    :param old_min: minimum of first range
    :type old_min: float
    :param old_max: maximum of first range
    :type old_max: float
    :param new_min: minimum of second range
    :type new_min: float
    :param new_max: maximum of second range
    :type new_max: float
    :param value: real number to remap
    :type value: float
    :return: remapped value
    :rtype: float
    """
    return new_min + ((value - old_min) * (new_max - new_min) / (old_max - old_min))