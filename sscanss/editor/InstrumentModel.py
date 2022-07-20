import os


class JsonAttribute:
    def __init__(self, mandatory=True, unique=False):
        self.mandatory = mandatory
        self.value = None

    def setValue(self, new_value):
        pass


class JsonString(JsonAttribute):
    def __init__(self, mandatory=True, unique=False):
        super().__init__(mandatory, unique)
        self.value = ''

    def setValue(self, new_value):
        self.value = new_value


class JsonFile(JsonAttribute):
    def __init__(self, mandatory=True, unique=False):
        super().__init__(mandatory, unique)
        self.value = ''

    def setValue(self, file_path):
        try:
            os.open(file_path)
        except (OSError, ValueError):
            raise ValueError("New value is not a valid filepath")


class JsonFloat(JsonAttribute):
    def __init__(self, mandatory=True, unique=False):
        super().__init__(mandatory, unique)
        self.value = 0.0

    def setValue(self, new_float):
        self.value = float(new_float)

class JsonFloatVec(JsonAttribute):
    def __init__(self, size, mandatory=True, unique=False):
        super().__init__(mandatory, unique)
        self.value = [0.0] * size

    def setValue(self, new_float, index):
        self.value[index] = float(new_float)

class JsonObjArray(JsonFloat):
    def __init__(self, mandatory=True, unique=False):
        super().__init__(mandatory, unique)
        self.value = []


class JsonInt(JsonAttribute):
    def __init__(self, mandatory=True, unique=False):
        super().__init__(mandatory, unique)
        self.value = 0

    def setValue(self, new_float):
        self.value = int(new_float)

class JsonObject:
    def __init__(self, name, attributes):
        self.name = name
        self.attributes = attributes

    def setAttribute(self, name, newValue):
        if self.attributes[name].type == type(newValue):
            self.attributes[name].value = newValue
        else:
            raise ValueError()

