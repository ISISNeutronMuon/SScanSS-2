VERSION = "2.14.2"
from fastjsonschema import JsonSchemaException


NoneType = type(None)

def validate(data):
    if not isinstance(data, (dict)):
        raise JsonSchemaException("data must be object", value=data, name="data", definition={'$schema': 'http://json-schema.org/draft-07/schema#', 'definitions': {'visual': {'$id': '#visual', 'type': 'object', 'properties': {'pose': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, 'colour': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'mesh': {'type': 'string'}}, 'required': ['mesh']}, 'joint': {'$id': '#joint', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'type': {'type': 'string', 'enum': ['prismatic', 'revolute']}, 'parent': {'type': 'string'}, 'child': {'type': 'string'}, 'axis': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'origin': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'lower_limit': {'type': 'number'}, 'upper_limit': {'type': 'number'}, 'home_offset': {'type': 'number'}}, 'required': ['name', 'type', 'parent', 'child', 'axis', 'origin', 'lower_limit', 'upper_limit']}, 'detector': {'$id': '#detector', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'default_collimator': {'type': 'string'}, 'diffracted_beam': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'positioner': {'type': 'string'}}, 'required': ['name', 'diffracted_beam']}, 'collimator': {'$id': '#collimator', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'detector': {'type': 'string'}, 'aperture': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['name', 'detector', 'visual']}, 'hardware': {'$id': '#hardware', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['name', 'visual']}, 'link': {'$id': '#link', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['name']}, 'positioner': {'$id': '#positioner', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'base': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, 'tool': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, 'custom_order': {'type': 'array', 'items': {'type': 'string'}}, 'joints': {'type': 'array', 'items': {'$ref': '#/definitions/joint'}}, 'links': {'type': 'array', 'items': {'$ref': '#/definitions/link'}}}, 'required': ['name', 'joints', 'links']}, 'positioning_stack': {'$id': '#positioning_stack', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'positioners': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['name', 'positioners']}}, 'title': 'Instrument', 'description': 'SScanSS-2 instrument', 'type': 'object', 'properties': {'instrument': {'type': 'object', 'properties': {'name': {'description': 'The unique name of the instrument', 'type': 'string'}, 'version': {'description': 'The version number of file', 'type': 'string'}, 'script_template': {'description': 'The path of script template', 'type': 'string'}, 'gauge_volume': {'description': 'The centre of gauge volume', 'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'incident_jaws': {'type': 'object', 'properties': {'aperture': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'aperture_lower_limit': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'aperture_upper_limit': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'beam_direction': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'beam_source': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'positioner': {'type': 'string'}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['aperture', 'aperture_lower_limit', 'aperture_upper_limit', 'beam_direction', 'beam_source', 'visual']}, 'detectors': {'type': 'array', 'items': {'$ref': '#/definitions/detector'}}, 'collimators': {'type': 'array', 'items': {'$ref': '#/definitions/collimator'}}, 'positioning_stacks': {'type': 'array', 'items': {'$ref': '#/definitions/positioning_stack'}}, 'positioners': {'type': 'array', 'items': {'$ref': '#/definitions/positioner'}}, 'fixed_hardware': {'type': 'array', 'items': {'$ref': '#/definitions/hardware'}}}, 'required': ['name', 'version', 'gauge_volume', 'incident_jaws', 'detectors', 'collimators', 'positioning_stacks', 'positioners']}}, 'required': ['instrument']}, rule='type')
    data_is_dict = isinstance(data, dict)
    if data_is_dict:
        data_len = len(data)
        if not all(prop in data for prop in ['instrument']):
            raise JsonSchemaException("data must contain ['instrument'] properties", value=data, name="data", definition={'$schema': 'http://json-schema.org/draft-07/schema#', 'definitions': {'visual': {'$id': '#visual', 'type': 'object', 'properties': {'pose': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, 'colour': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'mesh': {'type': 'string'}}, 'required': ['mesh']}, 'joint': {'$id': '#joint', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'type': {'type': 'string', 'enum': ['prismatic', 'revolute']}, 'parent': {'type': 'string'}, 'child': {'type': 'string'}, 'axis': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'origin': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'lower_limit': {'type': 'number'}, 'upper_limit': {'type': 'number'}, 'home_offset': {'type': 'number'}}, 'required': ['name', 'type', 'parent', 'child', 'axis', 'origin', 'lower_limit', 'upper_limit']}, 'detector': {'$id': '#detector', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'default_collimator': {'type': 'string'}, 'diffracted_beam': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'positioner': {'type': 'string'}}, 'required': ['name', 'diffracted_beam']}, 'collimator': {'$id': '#collimator', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'detector': {'type': 'string'}, 'aperture': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['name', 'detector', 'visual']}, 'hardware': {'$id': '#hardware', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['name', 'visual']}, 'link': {'$id': '#link', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['name']}, 'positioner': {'$id': '#positioner', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'base': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, 'tool': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, 'custom_order': {'type': 'array', 'items': {'type': 'string'}}, 'joints': {'type': 'array', 'items': {'$ref': '#/definitions/joint'}}, 'links': {'type': 'array', 'items': {'$ref': '#/definitions/link'}}}, 'required': ['name', 'joints', 'links']}, 'positioning_stack': {'$id': '#positioning_stack', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'positioners': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['name', 'positioners']}}, 'title': 'Instrument', 'description': 'SScanSS-2 instrument', 'type': 'object', 'properties': {'instrument': {'type': 'object', 'properties': {'name': {'description': 'The unique name of the instrument', 'type': 'string'}, 'version': {'description': 'The version number of file', 'type': 'string'}, 'script_template': {'description': 'The path of script template', 'type': 'string'}, 'gauge_volume': {'description': 'The centre of gauge volume', 'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'incident_jaws': {'type': 'object', 'properties': {'aperture': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'aperture_lower_limit': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'aperture_upper_limit': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'beam_direction': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'beam_source': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'positioner': {'type': 'string'}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['aperture', 'aperture_lower_limit', 'aperture_upper_limit', 'beam_direction', 'beam_source', 'visual']}, 'detectors': {'type': 'array', 'items': {'$ref': '#/definitions/detector'}}, 'collimators': {'type': 'array', 'items': {'$ref': '#/definitions/collimator'}}, 'positioning_stacks': {'type': 'array', 'items': {'$ref': '#/definitions/positioning_stack'}}, 'positioners': {'type': 'array', 'items': {'$ref': '#/definitions/positioner'}}, 'fixed_hardware': {'type': 'array', 'items': {'$ref': '#/definitions/hardware'}}}, 'required': ['name', 'version', 'gauge_volume', 'incident_jaws', 'detectors', 'collimators', 'positioning_stacks', 'positioners']}}, 'required': ['instrument']}, rule='required')
        data_keys = set(data.keys())
        if "instrument" in data_keys:
            data_keys.remove("instrument")
            data__instrument = data["instrument"]
            if not isinstance(data__instrument, (dict)):
                raise JsonSchemaException("data.instrument must be object", value=data__instrument, name="data.instrument", definition={'type': 'object', 'properties': {'name': {'description': 'The unique name of the instrument', 'type': 'string'}, 'version': {'description': 'The version number of file', 'type': 'string'}, 'script_template': {'description': 'The path of script template', 'type': 'string'}, 'gauge_volume': {'description': 'The centre of gauge volume', 'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'incident_jaws': {'type': 'object', 'properties': {'aperture': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'aperture_lower_limit': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'aperture_upper_limit': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'beam_direction': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'beam_source': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'positioner': {'type': 'string'}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['aperture', 'aperture_lower_limit', 'aperture_upper_limit', 'beam_direction', 'beam_source', 'visual']}, 'detectors': {'type': 'array', 'items': {'$ref': '#/definitions/detector'}}, 'collimators': {'type': 'array', 'items': {'$ref': '#/definitions/collimator'}}, 'positioning_stacks': {'type': 'array', 'items': {'$ref': '#/definitions/positioning_stack'}}, 'positioners': {'type': 'array', 'items': {'$ref': '#/definitions/positioner'}}, 'fixed_hardware': {'type': 'array', 'items': {'$ref': '#/definitions/hardware'}}}, 'required': ['name', 'version', 'gauge_volume', 'incident_jaws', 'detectors', 'collimators', 'positioning_stacks', 'positioners']}, rule='type')
            data__instrument_is_dict = isinstance(data__instrument, dict)
            if data__instrument_is_dict:
                data__instrument_len = len(data__instrument)
                if not all(prop in data__instrument for prop in ['name', 'version', 'gauge_volume', 'incident_jaws', 'detectors', 'collimators', 'positioning_stacks', 'positioners']):
                    raise JsonSchemaException("data.instrument must contain ['name', 'version', 'gauge_volume', 'incident_jaws', 'detectors', 'collimators', 'positioning_stacks', 'positioners'] properties", value=data__instrument, name="data.instrument", definition={'type': 'object', 'properties': {'name': {'description': 'The unique name of the instrument', 'type': 'string'}, 'version': {'description': 'The version number of file', 'type': 'string'}, 'script_template': {'description': 'The path of script template', 'type': 'string'}, 'gauge_volume': {'description': 'The centre of gauge volume', 'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'incident_jaws': {'type': 'object', 'properties': {'aperture': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'aperture_lower_limit': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'aperture_upper_limit': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'beam_direction': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'beam_source': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'positioner': {'type': 'string'}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['aperture', 'aperture_lower_limit', 'aperture_upper_limit', 'beam_direction', 'beam_source', 'visual']}, 'detectors': {'type': 'array', 'items': {'$ref': '#/definitions/detector'}}, 'collimators': {'type': 'array', 'items': {'$ref': '#/definitions/collimator'}}, 'positioning_stacks': {'type': 'array', 'items': {'$ref': '#/definitions/positioning_stack'}}, 'positioners': {'type': 'array', 'items': {'$ref': '#/definitions/positioner'}}, 'fixed_hardware': {'type': 'array', 'items': {'$ref': '#/definitions/hardware'}}}, 'required': ['name', 'version', 'gauge_volume', 'incident_jaws', 'detectors', 'collimators', 'positioning_stacks', 'positioners']}, rule='required')
                data__instrument_keys = set(data__instrument.keys())
                if "name" in data__instrument_keys:
                    data__instrument_keys.remove("name")
                    data__instrument__name = data__instrument["name"]
                    if not isinstance(data__instrument__name, (str)):
                        raise JsonSchemaException("data.instrument.name must be string", value=data__instrument__name, name="data.instrument.name", definition={'description': 'The unique name of the instrument', 'type': 'string'}, rule='type')
                if "version" in data__instrument_keys:
                    data__instrument_keys.remove("version")
                    data__instrument__version = data__instrument["version"]
                    if not isinstance(data__instrument__version, (str)):
                        raise JsonSchemaException("data.instrument.version must be string", value=data__instrument__version, name="data.instrument.version", definition={'description': 'The version number of file', 'type': 'string'}, rule='type')
                if "script_template" in data__instrument_keys:
                    data__instrument_keys.remove("script_template")
                    data__instrument__scripttemplate = data__instrument["script_template"]
                    if not isinstance(data__instrument__scripttemplate, (str)):
                        raise JsonSchemaException("data.instrument.script_template must be string", value=data__instrument__scripttemplate, name="data.instrument.script_template", definition={'description': 'The path of script template', 'type': 'string'}, rule='type')
                if "gauge_volume" in data__instrument_keys:
                    data__instrument_keys.remove("gauge_volume")
                    data__instrument__gaugevolume = data__instrument["gauge_volume"]
                    if not isinstance(data__instrument__gaugevolume, (list)):
                        raise JsonSchemaException("data.instrument.gauge_volume must be array", value=data__instrument__gaugevolume, name="data.instrument.gauge_volume", definition={'description': 'The centre of gauge volume', 'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='type')
                    data__instrument__gaugevolume_is_list = isinstance(data__instrument__gaugevolume, list)
                    if data__instrument__gaugevolume_is_list:
                        data__instrument__gaugevolume_len = len(data__instrument__gaugevolume)
                        if data__instrument__gaugevolume_len < 3:
                            raise JsonSchemaException("data.instrument.gauge_volume must contain at least 3 items", value=data__instrument__gaugevolume, name="data.instrument.gauge_volume", definition={'description': 'The centre of gauge volume', 'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='minItems')
                        if data__instrument__gaugevolume_len > 3:
                            raise JsonSchemaException("data.instrument.gauge_volume must contain less than or equal to 3 items", value=data__instrument__gaugevolume, name="data.instrument.gauge_volume", definition={'description': 'The centre of gauge volume', 'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='maxItems')
                        for data__instrument__gaugevolume_x, data__instrument__gaugevolume_item in enumerate(data__instrument__gaugevolume):
                            if not isinstance(data__instrument__gaugevolume_item, (int, float)) or isinstance(data__instrument__gaugevolume_item, bool):
                                raise JsonSchemaException(""+"data.instrument.gauge_volume[{data__instrument__gaugevolume_x}]".format(**locals())+" must be number", value=data__instrument__gaugevolume_item, name=""+"data.instrument.gauge_volume[{data__instrument__gaugevolume_x}]".format(**locals())+"", definition={'type': 'number'}, rule='type')
                if "incident_jaws" in data__instrument_keys:
                    data__instrument_keys.remove("incident_jaws")
                    data__instrument__incidentjaws = data__instrument["incident_jaws"]
                    if not isinstance(data__instrument__incidentjaws, (dict)):
                        raise JsonSchemaException("data.instrument.incident_jaws must be object", value=data__instrument__incidentjaws, name="data.instrument.incident_jaws", definition={'type': 'object', 'properties': {'aperture': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'aperture_lower_limit': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'aperture_upper_limit': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'beam_direction': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'beam_source': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'positioner': {'type': 'string'}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['aperture', 'aperture_lower_limit', 'aperture_upper_limit', 'beam_direction', 'beam_source', 'visual']}, rule='type')
                    data__instrument__incidentjaws_is_dict = isinstance(data__instrument__incidentjaws, dict)
                    if data__instrument__incidentjaws_is_dict:
                        data__instrument__incidentjaws_len = len(data__instrument__incidentjaws)
                        if not all(prop in data__instrument__incidentjaws for prop in ['aperture', 'aperture_lower_limit', 'aperture_upper_limit', 'beam_direction', 'beam_source', 'visual']):
                            raise JsonSchemaException("data.instrument.incident_jaws must contain ['aperture', 'aperture_lower_limit', 'aperture_upper_limit', 'beam_direction', 'beam_source', 'visual'] properties", value=data__instrument__incidentjaws, name="data.instrument.incident_jaws", definition={'type': 'object', 'properties': {'aperture': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'aperture_lower_limit': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'aperture_upper_limit': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'beam_direction': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'beam_source': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'positioner': {'type': 'string'}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['aperture', 'aperture_lower_limit', 'aperture_upper_limit', 'beam_direction', 'beam_source', 'visual']}, rule='required')
                        data__instrument__incidentjaws_keys = set(data__instrument__incidentjaws.keys())
                        if "aperture" in data__instrument__incidentjaws_keys:
                            data__instrument__incidentjaws_keys.remove("aperture")
                            data__instrument__incidentjaws__aperture = data__instrument__incidentjaws["aperture"]
                            if not isinstance(data__instrument__incidentjaws__aperture, (list)):
                                raise JsonSchemaException("data.instrument.incident_jaws.aperture must be array", value=data__instrument__incidentjaws__aperture, name="data.instrument.incident_jaws.aperture", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, rule='type')
                            data__instrument__incidentjaws__aperture_is_list = isinstance(data__instrument__incidentjaws__aperture, list)
                            if data__instrument__incidentjaws__aperture_is_list:
                                data__instrument__incidentjaws__aperture_len = len(data__instrument__incidentjaws__aperture)
                                if data__instrument__incidentjaws__aperture_len < 2:
                                    raise JsonSchemaException("data.instrument.incident_jaws.aperture must contain at least 2 items", value=data__instrument__incidentjaws__aperture, name="data.instrument.incident_jaws.aperture", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, rule='minItems')
                                if data__instrument__incidentjaws__aperture_len > 2:
                                    raise JsonSchemaException("data.instrument.incident_jaws.aperture must contain less than or equal to 2 items", value=data__instrument__incidentjaws__aperture, name="data.instrument.incident_jaws.aperture", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, rule='maxItems')
                                for data__instrument__incidentjaws__aperture_x, data__instrument__incidentjaws__aperture_item in enumerate(data__instrument__incidentjaws__aperture):
                                    if not isinstance(data__instrument__incidentjaws__aperture_item, (int, float)) or isinstance(data__instrument__incidentjaws__aperture_item, bool):
                                        raise JsonSchemaException(""+"data.instrument.incident_jaws.aperture[{data__instrument__incidentjaws__aperture_x}]".format(**locals())+" must be number", value=data__instrument__incidentjaws__aperture_item, name=""+"data.instrument.incident_jaws.aperture[{data__instrument__incidentjaws__aperture_x}]".format(**locals())+"", definition={'type': 'number'}, rule='type')
                        if "aperture_lower_limit" in data__instrument__incidentjaws_keys:
                            data__instrument__incidentjaws_keys.remove("aperture_lower_limit")
                            data__instrument__incidentjaws__aperturelowerlimit = data__instrument__incidentjaws["aperture_lower_limit"]
                            if not isinstance(data__instrument__incidentjaws__aperturelowerlimit, (list)):
                                raise JsonSchemaException("data.instrument.incident_jaws.aperture_lower_limit must be array", value=data__instrument__incidentjaws__aperturelowerlimit, name="data.instrument.incident_jaws.aperture_lower_limit", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, rule='type')
                            data__instrument__incidentjaws__aperturelowerlimit_is_list = isinstance(data__instrument__incidentjaws__aperturelowerlimit, list)
                            if data__instrument__incidentjaws__aperturelowerlimit_is_list:
                                data__instrument__incidentjaws__aperturelowerlimit_len = len(data__instrument__incidentjaws__aperturelowerlimit)
                                if data__instrument__incidentjaws__aperturelowerlimit_len < 2:
                                    raise JsonSchemaException("data.instrument.incident_jaws.aperture_lower_limit must contain at least 2 items", value=data__instrument__incidentjaws__aperturelowerlimit, name="data.instrument.incident_jaws.aperture_lower_limit", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, rule='minItems')
                                if data__instrument__incidentjaws__aperturelowerlimit_len > 2:
                                    raise JsonSchemaException("data.instrument.incident_jaws.aperture_lower_limit must contain less than or equal to 2 items", value=data__instrument__incidentjaws__aperturelowerlimit, name="data.instrument.incident_jaws.aperture_lower_limit", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, rule='maxItems')
                                for data__instrument__incidentjaws__aperturelowerlimit_x, data__instrument__incidentjaws__aperturelowerlimit_item in enumerate(data__instrument__incidentjaws__aperturelowerlimit):
                                    if not isinstance(data__instrument__incidentjaws__aperturelowerlimit_item, (int, float)) or isinstance(data__instrument__incidentjaws__aperturelowerlimit_item, bool):
                                        raise JsonSchemaException(""+"data.instrument.incident_jaws.aperture_lower_limit[{data__instrument__incidentjaws__aperturelowerlimit_x}]".format(**locals())+" must be number", value=data__instrument__incidentjaws__aperturelowerlimit_item, name=""+"data.instrument.incident_jaws.aperture_lower_limit[{data__instrument__incidentjaws__aperturelowerlimit_x}]".format(**locals())+"", definition={'type': 'number'}, rule='type')
                        if "aperture_upper_limit" in data__instrument__incidentjaws_keys:
                            data__instrument__incidentjaws_keys.remove("aperture_upper_limit")
                            data__instrument__incidentjaws__apertureupperlimit = data__instrument__incidentjaws["aperture_upper_limit"]
                            if not isinstance(data__instrument__incidentjaws__apertureupperlimit, (list)):
                                raise JsonSchemaException("data.instrument.incident_jaws.aperture_upper_limit must be array", value=data__instrument__incidentjaws__apertureupperlimit, name="data.instrument.incident_jaws.aperture_upper_limit", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, rule='type')
                            data__instrument__incidentjaws__apertureupperlimit_is_list = isinstance(data__instrument__incidentjaws__apertureupperlimit, list)
                            if data__instrument__incidentjaws__apertureupperlimit_is_list:
                                data__instrument__incidentjaws__apertureupperlimit_len = len(data__instrument__incidentjaws__apertureupperlimit)
                                if data__instrument__incidentjaws__apertureupperlimit_len < 2:
                                    raise JsonSchemaException("data.instrument.incident_jaws.aperture_upper_limit must contain at least 2 items", value=data__instrument__incidentjaws__apertureupperlimit, name="data.instrument.incident_jaws.aperture_upper_limit", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, rule='minItems')
                                if data__instrument__incidentjaws__apertureupperlimit_len > 2:
                                    raise JsonSchemaException("data.instrument.incident_jaws.aperture_upper_limit must contain less than or equal to 2 items", value=data__instrument__incidentjaws__apertureupperlimit, name="data.instrument.incident_jaws.aperture_upper_limit", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, rule='maxItems')
                                for data__instrument__incidentjaws__apertureupperlimit_x, data__instrument__incidentjaws__apertureupperlimit_item in enumerate(data__instrument__incidentjaws__apertureupperlimit):
                                    if not isinstance(data__instrument__incidentjaws__apertureupperlimit_item, (int, float)) or isinstance(data__instrument__incidentjaws__apertureupperlimit_item, bool):
                                        raise JsonSchemaException(""+"data.instrument.incident_jaws.aperture_upper_limit[{data__instrument__incidentjaws__apertureupperlimit_x}]".format(**locals())+" must be number", value=data__instrument__incidentjaws__apertureupperlimit_item, name=""+"data.instrument.incident_jaws.aperture_upper_limit[{data__instrument__incidentjaws__apertureupperlimit_x}]".format(**locals())+"", definition={'type': 'number'}, rule='type')
                        if "beam_direction" in data__instrument__incidentjaws_keys:
                            data__instrument__incidentjaws_keys.remove("beam_direction")
                            data__instrument__incidentjaws__beamdirection = data__instrument__incidentjaws["beam_direction"]
                            if not isinstance(data__instrument__incidentjaws__beamdirection, (list)):
                                raise JsonSchemaException("data.instrument.incident_jaws.beam_direction must be array", value=data__instrument__incidentjaws__beamdirection, name="data.instrument.incident_jaws.beam_direction", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='type')
                            data__instrument__incidentjaws__beamdirection_is_list = isinstance(data__instrument__incidentjaws__beamdirection, list)
                            if data__instrument__incidentjaws__beamdirection_is_list:
                                data__instrument__incidentjaws__beamdirection_len = len(data__instrument__incidentjaws__beamdirection)
                                if data__instrument__incidentjaws__beamdirection_len < 3:
                                    raise JsonSchemaException("data.instrument.incident_jaws.beam_direction must contain at least 3 items", value=data__instrument__incidentjaws__beamdirection, name="data.instrument.incident_jaws.beam_direction", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='minItems')
                                if data__instrument__incidentjaws__beamdirection_len > 3:
                                    raise JsonSchemaException("data.instrument.incident_jaws.beam_direction must contain less than or equal to 3 items", value=data__instrument__incidentjaws__beamdirection, name="data.instrument.incident_jaws.beam_direction", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='maxItems')
                                for data__instrument__incidentjaws__beamdirection_x, data__instrument__incidentjaws__beamdirection_item in enumerate(data__instrument__incidentjaws__beamdirection):
                                    if not isinstance(data__instrument__incidentjaws__beamdirection_item, (int, float)) or isinstance(data__instrument__incidentjaws__beamdirection_item, bool):
                                        raise JsonSchemaException(""+"data.instrument.incident_jaws.beam_direction[{data__instrument__incidentjaws__beamdirection_x}]".format(**locals())+" must be number", value=data__instrument__incidentjaws__beamdirection_item, name=""+"data.instrument.incident_jaws.beam_direction[{data__instrument__incidentjaws__beamdirection_x}]".format(**locals())+"", definition={'type': 'number'}, rule='type')
                        if "beam_source" in data__instrument__incidentjaws_keys:
                            data__instrument__incidentjaws_keys.remove("beam_source")
                            data__instrument__incidentjaws__beamsource = data__instrument__incidentjaws["beam_source"]
                            if not isinstance(data__instrument__incidentjaws__beamsource, (list)):
                                raise JsonSchemaException("data.instrument.incident_jaws.beam_source must be array", value=data__instrument__incidentjaws__beamsource, name="data.instrument.incident_jaws.beam_source", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='type')
                            data__instrument__incidentjaws__beamsource_is_list = isinstance(data__instrument__incidentjaws__beamsource, list)
                            if data__instrument__incidentjaws__beamsource_is_list:
                                data__instrument__incidentjaws__beamsource_len = len(data__instrument__incidentjaws__beamsource)
                                if data__instrument__incidentjaws__beamsource_len < 3:
                                    raise JsonSchemaException("data.instrument.incident_jaws.beam_source must contain at least 3 items", value=data__instrument__incidentjaws__beamsource, name="data.instrument.incident_jaws.beam_source", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='minItems')
                                if data__instrument__incidentjaws__beamsource_len > 3:
                                    raise JsonSchemaException("data.instrument.incident_jaws.beam_source must contain less than or equal to 3 items", value=data__instrument__incidentjaws__beamsource, name="data.instrument.incident_jaws.beam_source", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='maxItems')
                                for data__instrument__incidentjaws__beamsource_x, data__instrument__incidentjaws__beamsource_item in enumerate(data__instrument__incidentjaws__beamsource):
                                    if not isinstance(data__instrument__incidentjaws__beamsource_item, (int, float)) or isinstance(data__instrument__incidentjaws__beamsource_item, bool):
                                        raise JsonSchemaException(""+"data.instrument.incident_jaws.beam_source[{data__instrument__incidentjaws__beamsource_x}]".format(**locals())+" must be number", value=data__instrument__incidentjaws__beamsource_item, name=""+"data.instrument.incident_jaws.beam_source[{data__instrument__incidentjaws__beamsource_x}]".format(**locals())+"", definition={'type': 'number'}, rule='type')
                        if "positioner" in data__instrument__incidentjaws_keys:
                            data__instrument__incidentjaws_keys.remove("positioner")
                            data__instrument__incidentjaws__positioner = data__instrument__incidentjaws["positioner"]
                            if not isinstance(data__instrument__incidentjaws__positioner, (str)):
                                raise JsonSchemaException("data.instrument.incident_jaws.positioner must be string", value=data__instrument__incidentjaws__positioner, name="data.instrument.incident_jaws.positioner", definition={'type': 'string'}, rule='type')
                        if "visual" in data__instrument__incidentjaws_keys:
                            data__instrument__incidentjaws_keys.remove("visual")
                            data__instrument__incidentjaws__visual = data__instrument__incidentjaws["visual"]
                            validate___definitions_visual(data__instrument__incidentjaws__visual)
                if "detectors" in data__instrument_keys:
                    data__instrument_keys.remove("detectors")
                    data__instrument__detectors = data__instrument["detectors"]
                    if not isinstance(data__instrument__detectors, (list)):
                        raise JsonSchemaException("data.instrument.detectors must be array", value=data__instrument__detectors, name="data.instrument.detectors", definition={'type': 'array', 'items': {'$ref': '#/definitions/detector'}}, rule='type')
                    data__instrument__detectors_is_list = isinstance(data__instrument__detectors, list)
                    if data__instrument__detectors_is_list:
                        data__instrument__detectors_len = len(data__instrument__detectors)
                        for data__instrument__detectors_x, data__instrument__detectors_item in enumerate(data__instrument__detectors):
                            validate___definitions_detector(data__instrument__detectors_item)
                if "collimators" in data__instrument_keys:
                    data__instrument_keys.remove("collimators")
                    data__instrument__collimators = data__instrument["collimators"]
                    if not isinstance(data__instrument__collimators, (list)):
                        raise JsonSchemaException("data.instrument.collimators must be array", value=data__instrument__collimators, name="data.instrument.collimators", definition={'type': 'array', 'items': {'$ref': '#/definitions/collimator'}}, rule='type')
                    data__instrument__collimators_is_list = isinstance(data__instrument__collimators, list)
                    if data__instrument__collimators_is_list:
                        data__instrument__collimators_len = len(data__instrument__collimators)
                        for data__instrument__collimators_x, data__instrument__collimators_item in enumerate(data__instrument__collimators):
                            validate___definitions_collimator(data__instrument__collimators_item)
                if "positioning_stacks" in data__instrument_keys:
                    data__instrument_keys.remove("positioning_stacks")
                    data__instrument__positioningstacks = data__instrument["positioning_stacks"]
                    if not isinstance(data__instrument__positioningstacks, (list)):
                        raise JsonSchemaException("data.instrument.positioning_stacks must be array", value=data__instrument__positioningstacks, name="data.instrument.positioning_stacks", definition={'type': 'array', 'items': {'$ref': '#/definitions/positioning_stack'}}, rule='type')
                    data__instrument__positioningstacks_is_list = isinstance(data__instrument__positioningstacks, list)
                    if data__instrument__positioningstacks_is_list:
                        data__instrument__positioningstacks_len = len(data__instrument__positioningstacks)
                        for data__instrument__positioningstacks_x, data__instrument__positioningstacks_item in enumerate(data__instrument__positioningstacks):
                            validate___definitions_positioning_stack(data__instrument__positioningstacks_item)
                if "positioners" in data__instrument_keys:
                    data__instrument_keys.remove("positioners")
                    data__instrument__positioners = data__instrument["positioners"]
                    if not isinstance(data__instrument__positioners, (list)):
                        raise JsonSchemaException("data.instrument.positioners must be array", value=data__instrument__positioners, name="data.instrument.positioners", definition={'type': 'array', 'items': {'$ref': '#/definitions/positioner'}}, rule='type')
                    data__instrument__positioners_is_list = isinstance(data__instrument__positioners, list)
                    if data__instrument__positioners_is_list:
                        data__instrument__positioners_len = len(data__instrument__positioners)
                        for data__instrument__positioners_x, data__instrument__positioners_item in enumerate(data__instrument__positioners):
                            validate___definitions_positioner(data__instrument__positioners_item)
                if "fixed_hardware" in data__instrument_keys:
                    data__instrument_keys.remove("fixed_hardware")
                    data__instrument__fixedhardware = data__instrument["fixed_hardware"]
                    if not isinstance(data__instrument__fixedhardware, (list)):
                        raise JsonSchemaException("data.instrument.fixed_hardware must be array", value=data__instrument__fixedhardware, name="data.instrument.fixed_hardware", definition={'type': 'array', 'items': {'$ref': '#/definitions/hardware'}}, rule='type')
                    data__instrument__fixedhardware_is_list = isinstance(data__instrument__fixedhardware, list)
                    if data__instrument__fixedhardware_is_list:
                        data__instrument__fixedhardware_len = len(data__instrument__fixedhardware)
                        for data__instrument__fixedhardware_x, data__instrument__fixedhardware_item in enumerate(data__instrument__fixedhardware):
                            validate___definitions_hardware(data__instrument__fixedhardware_item)
    return data

def validate___definitions_hardware(data):
    if not isinstance(data, (dict)):
        raise JsonSchemaException("data must be object", value=data, name="data", definition={'$id': '#hardware', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['name', 'visual']}, rule='type')
    data_is_dict = isinstance(data, dict)
    if data_is_dict:
        data_len = len(data)
        if not all(prop in data for prop in ['name', 'visual']):
            raise JsonSchemaException("data must contain ['name', 'visual'] properties", value=data, name="data", definition={'$id': '#hardware', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['name', 'visual']}, rule='required')
        data_keys = set(data.keys())
        if "name" in data_keys:
            data_keys.remove("name")
            data__name = data["name"]
            if not isinstance(data__name, (str)):
                raise JsonSchemaException("data.name must be string", value=data__name, name="data.name", definition={'type': 'string'}, rule='type')
        if "visual" in data_keys:
            data_keys.remove("visual")
            data__visual = data["visual"]
            validate___definitions_visual(data__visual)
    return data

def validate___definitions_positioner(data):
    if not isinstance(data, (dict)):
        raise JsonSchemaException("data must be object", value=data, name="data", definition={'$id': '#positioner', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'base': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, 'tool': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, 'custom_order': {'type': 'array', 'items': {'type': 'string'}}, 'joints': {'type': 'array', 'items': {'$ref': '#/definitions/joint'}}, 'links': {'type': 'array', 'items': {'$ref': '#/definitions/link'}}}, 'required': ['name', 'joints', 'links']}, rule='type')
    data_is_dict = isinstance(data, dict)
    if data_is_dict:
        data_len = len(data)
        if not all(prop in data for prop in ['name', 'joints', 'links']):
            raise JsonSchemaException("data must contain ['name', 'joints', 'links'] properties", value=data, name="data", definition={'$id': '#positioner', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'base': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, 'tool': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, 'custom_order': {'type': 'array', 'items': {'type': 'string'}}, 'joints': {'type': 'array', 'items': {'$ref': '#/definitions/joint'}}, 'links': {'type': 'array', 'items': {'$ref': '#/definitions/link'}}}, 'required': ['name', 'joints', 'links']}, rule='required')
        data_keys = set(data.keys())
        if "name" in data_keys:
            data_keys.remove("name")
            data__name = data["name"]
            if not isinstance(data__name, (str)):
                raise JsonSchemaException("data.name must be string", value=data__name, name="data.name", definition={'type': 'string'}, rule='type')
        if "base" in data_keys:
            data_keys.remove("base")
            data__base = data["base"]
            if not isinstance(data__base, (list)):
                raise JsonSchemaException("data.base must be array", value=data__base, name="data.base", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, rule='type')
            data__base_is_list = isinstance(data__base, list)
            if data__base_is_list:
                data__base_len = len(data__base)
                if data__base_len < 6:
                    raise JsonSchemaException("data.base must contain at least 6 items", value=data__base, name="data.base", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, rule='minItems')
                if data__base_len > 6:
                    raise JsonSchemaException("data.base must contain less than or equal to 6 items", value=data__base, name="data.base", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, rule='maxItems')
                for data__base_x, data__base_item in enumerate(data__base):
                    if not isinstance(data__base_item, (int, float)) or isinstance(data__base_item, bool):
                        raise JsonSchemaException(""+"data.base[{data__base_x}]".format(**locals())+" must be number", value=data__base_item, name=""+"data.base[{data__base_x}]".format(**locals())+"", definition={'type': 'number'}, rule='type')
        if "tool" in data_keys:
            data_keys.remove("tool")
            data__tool = data["tool"]
            if not isinstance(data__tool, (list)):
                raise JsonSchemaException("data.tool must be array", value=data__tool, name="data.tool", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, rule='type')
            data__tool_is_list = isinstance(data__tool, list)
            if data__tool_is_list:
                data__tool_len = len(data__tool)
                if data__tool_len < 6:
                    raise JsonSchemaException("data.tool must contain at least 6 items", value=data__tool, name="data.tool", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, rule='minItems')
                if data__tool_len > 6:
                    raise JsonSchemaException("data.tool must contain less than or equal to 6 items", value=data__tool, name="data.tool", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, rule='maxItems')
                for data__tool_x, data__tool_item in enumerate(data__tool):
                    if not isinstance(data__tool_item, (int, float)) or isinstance(data__tool_item, bool):
                        raise JsonSchemaException(""+"data.tool[{data__tool_x}]".format(**locals())+" must be number", value=data__tool_item, name=""+"data.tool[{data__tool_x}]".format(**locals())+"", definition={'type': 'number'}, rule='type')
        if "custom_order" in data_keys:
            data_keys.remove("custom_order")
            data__customorder = data["custom_order"]
            if not isinstance(data__customorder, (list)):
                raise JsonSchemaException("data.custom_order must be array", value=data__customorder, name="data.custom_order", definition={'type': 'array', 'items': {'type': 'string'}}, rule='type')
            data__customorder_is_list = isinstance(data__customorder, list)
            if data__customorder_is_list:
                data__customorder_len = len(data__customorder)
                for data__customorder_x, data__customorder_item in enumerate(data__customorder):
                    if not isinstance(data__customorder_item, (str)):
                        raise JsonSchemaException(""+"data.custom_order[{data__customorder_x}]".format(**locals())+" must be string", value=data__customorder_item, name=""+"data.custom_order[{data__customorder_x}]".format(**locals())+"", definition={'type': 'string'}, rule='type')
        if "joints" in data_keys:
            data_keys.remove("joints")
            data__joints = data["joints"]
            if not isinstance(data__joints, (list)):
                raise JsonSchemaException("data.joints must be array", value=data__joints, name="data.joints", definition={'type': 'array', 'items': {'$ref': '#/definitions/joint'}}, rule='type')
            data__joints_is_list = isinstance(data__joints, list)
            if data__joints_is_list:
                data__joints_len = len(data__joints)
                for data__joints_x, data__joints_item in enumerate(data__joints):
                    validate___definitions_joint(data__joints_item)
        if "links" in data_keys:
            data_keys.remove("links")
            data__links = data["links"]
            if not isinstance(data__links, (list)):
                raise JsonSchemaException("data.links must be array", value=data__links, name="data.links", definition={'type': 'array', 'items': {'$ref': '#/definitions/link'}}, rule='type')
            data__links_is_list = isinstance(data__links, list)
            if data__links_is_list:
                data__links_len = len(data__links)
                for data__links_x, data__links_item in enumerate(data__links):
                    validate___definitions_link(data__links_item)
    return data

def validate___definitions_link(data):
    if not isinstance(data, (dict)):
        raise JsonSchemaException("data must be object", value=data, name="data", definition={'$id': '#link', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['name']}, rule='type')
    data_is_dict = isinstance(data, dict)
    if data_is_dict:
        data_len = len(data)
        if not all(prop in data for prop in ['name']):
            raise JsonSchemaException("data must contain ['name'] properties", value=data, name="data", definition={'$id': '#link', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['name']}, rule='required')
        data_keys = set(data.keys())
        if "name" in data_keys:
            data_keys.remove("name")
            data__name = data["name"]
            if not isinstance(data__name, (str)):
                raise JsonSchemaException("data.name must be string", value=data__name, name="data.name", definition={'type': 'string'}, rule='type')
        if "visual" in data_keys:
            data_keys.remove("visual")
            data__visual = data["visual"]
            validate___definitions_visual(data__visual)
    return data

def validate___definitions_joint(data):
    if not isinstance(data, (dict)):
        raise JsonSchemaException("data must be object", value=data, name="data", definition={'$id': '#joint', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'type': {'type': 'string', 'enum': ['prismatic', 'revolute']}, 'parent': {'type': 'string'}, 'child': {'type': 'string'}, 'axis': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'origin': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'lower_limit': {'type': 'number'}, 'upper_limit': {'type': 'number'}, 'home_offset': {'type': 'number'}}, 'required': ['name', 'type', 'parent', 'child', 'axis', 'origin', 'lower_limit', 'upper_limit']}, rule='type')
    data_is_dict = isinstance(data, dict)
    if data_is_dict:
        data_len = len(data)
        if not all(prop in data for prop in ['name', 'type', 'parent', 'child', 'axis', 'origin', 'lower_limit', 'upper_limit']):
            raise JsonSchemaException("data must contain ['name', 'type', 'parent', 'child', 'axis', 'origin', 'lower_limit', 'upper_limit'] properties", value=data, name="data", definition={'$id': '#joint', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'type': {'type': 'string', 'enum': ['prismatic', 'revolute']}, 'parent': {'type': 'string'}, 'child': {'type': 'string'}, 'axis': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'origin': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'lower_limit': {'type': 'number'}, 'upper_limit': {'type': 'number'}, 'home_offset': {'type': 'number'}}, 'required': ['name', 'type', 'parent', 'child', 'axis', 'origin', 'lower_limit', 'upper_limit']}, rule='required')
        data_keys = set(data.keys())
        if "name" in data_keys:
            data_keys.remove("name")
            data__name = data["name"]
            if not isinstance(data__name, (str)):
                raise JsonSchemaException("data.name must be string", value=data__name, name="data.name", definition={'type': 'string'}, rule='type')
        if "type" in data_keys:
            data_keys.remove("type")
            data__type = data["type"]
            if not isinstance(data__type, (str)):
                raise JsonSchemaException("data.type must be string", value=data__type, name="data.type", definition={'type': 'string', 'enum': ['prismatic', 'revolute']}, rule='type')
            if data__type not in ['prismatic', 'revolute']:
                raise JsonSchemaException("data.type must be one of ['prismatic', 'revolute']", value=data__type, name="data.type", definition={'type': 'string', 'enum': ['prismatic', 'revolute']}, rule='enum')
        if "parent" in data_keys:
            data_keys.remove("parent")
            data__parent = data["parent"]
            if not isinstance(data__parent, (str)):
                raise JsonSchemaException("data.parent must be string", value=data__parent, name="data.parent", definition={'type': 'string'}, rule='type')
        if "child" in data_keys:
            data_keys.remove("child")
            data__child = data["child"]
            if not isinstance(data__child, (str)):
                raise JsonSchemaException("data.child must be string", value=data__child, name="data.child", definition={'type': 'string'}, rule='type')
        if "axis" in data_keys:
            data_keys.remove("axis")
            data__axis = data["axis"]
            if not isinstance(data__axis, (list)):
                raise JsonSchemaException("data.axis must be array", value=data__axis, name="data.axis", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='type')
            data__axis_is_list = isinstance(data__axis, list)
            if data__axis_is_list:
                data__axis_len = len(data__axis)
                if data__axis_len < 3:
                    raise JsonSchemaException("data.axis must contain at least 3 items", value=data__axis, name="data.axis", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='minItems')
                if data__axis_len > 3:
                    raise JsonSchemaException("data.axis must contain less than or equal to 3 items", value=data__axis, name="data.axis", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='maxItems')
                for data__axis_x, data__axis_item in enumerate(data__axis):
                    if not isinstance(data__axis_item, (int, float)) or isinstance(data__axis_item, bool):
                        raise JsonSchemaException(""+"data.axis[{data__axis_x}]".format(**locals())+" must be number", value=data__axis_item, name=""+"data.axis[{data__axis_x}]".format(**locals())+"", definition={'type': 'number'}, rule='type')
        if "origin" in data_keys:
            data_keys.remove("origin")
            data__origin = data["origin"]
            if not isinstance(data__origin, (list)):
                raise JsonSchemaException("data.origin must be array", value=data__origin, name="data.origin", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='type')
            data__origin_is_list = isinstance(data__origin, list)
            if data__origin_is_list:
                data__origin_len = len(data__origin)
                if data__origin_len < 3:
                    raise JsonSchemaException("data.origin must contain at least 3 items", value=data__origin, name="data.origin", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='minItems')
                if data__origin_len > 3:
                    raise JsonSchemaException("data.origin must contain less than or equal to 3 items", value=data__origin, name="data.origin", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='maxItems')
                for data__origin_x, data__origin_item in enumerate(data__origin):
                    if not isinstance(data__origin_item, (int, float)) or isinstance(data__origin_item, bool):
                        raise JsonSchemaException(""+"data.origin[{data__origin_x}]".format(**locals())+" must be number", value=data__origin_item, name=""+"data.origin[{data__origin_x}]".format(**locals())+"", definition={'type': 'number'}, rule='type')
        if "lower_limit" in data_keys:
            data_keys.remove("lower_limit")
            data__lowerlimit = data["lower_limit"]
            if not isinstance(data__lowerlimit, (int, float)) or isinstance(data__lowerlimit, bool):
                raise JsonSchemaException("data.lower_limit must be number", value=data__lowerlimit, name="data.lower_limit", definition={'type': 'number'}, rule='type')
        if "upper_limit" in data_keys:
            data_keys.remove("upper_limit")
            data__upperlimit = data["upper_limit"]
            if not isinstance(data__upperlimit, (int, float)) or isinstance(data__upperlimit, bool):
                raise JsonSchemaException("data.upper_limit must be number", value=data__upperlimit, name="data.upper_limit", definition={'type': 'number'}, rule='type')
        if "home_offset" in data_keys:
            data_keys.remove("home_offset")
            data__homeoffset = data["home_offset"]
            if not isinstance(data__homeoffset, (int, float)) or isinstance(data__homeoffset, bool):
                raise JsonSchemaException("data.home_offset must be number", value=data__homeoffset, name="data.home_offset", definition={'type': 'number'}, rule='type')
    return data

def validate___definitions_positioning_stack(data):
    if not isinstance(data, (dict)):
        raise JsonSchemaException("data must be object", value=data, name="data", definition={'$id': '#positioning_stack', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'positioners': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['name', 'positioners']}, rule='type')
    data_is_dict = isinstance(data, dict)
    if data_is_dict:
        data_len = len(data)
        if not all(prop in data for prop in ['name', 'positioners']):
            raise JsonSchemaException("data must contain ['name', 'positioners'] properties", value=data, name="data", definition={'$id': '#positioning_stack', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'positioners': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['name', 'positioners']}, rule='required')
        data_keys = set(data.keys())
        if "name" in data_keys:
            data_keys.remove("name")
            data__name = data["name"]
            if not isinstance(data__name, (str)):
                raise JsonSchemaException("data.name must be string", value=data__name, name="data.name", definition={'type': 'string'}, rule='type')
        if "positioners" in data_keys:
            data_keys.remove("positioners")
            data__positioners = data["positioners"]
            if not isinstance(data__positioners, (list)):
                raise JsonSchemaException("data.positioners must be array", value=data__positioners, name="data.positioners", definition={'type': 'array', 'items': {'type': 'string'}}, rule='type')
            data__positioners_is_list = isinstance(data__positioners, list)
            if data__positioners_is_list:
                data__positioners_len = len(data__positioners)
                for data__positioners_x, data__positioners_item in enumerate(data__positioners):
                    if not isinstance(data__positioners_item, (str)):
                        raise JsonSchemaException(""+"data.positioners[{data__positioners_x}]".format(**locals())+" must be string", value=data__positioners_item, name=""+"data.positioners[{data__positioners_x}]".format(**locals())+"", definition={'type': 'string'}, rule='type')
    return data

def validate___definitions_collimator(data):
    if not isinstance(data, (dict)):
        raise JsonSchemaException("data must be object", value=data, name="data", definition={'$id': '#collimator', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'detector': {'type': 'string'}, 'aperture': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['name', 'detector', 'visual']}, rule='type')
    data_is_dict = isinstance(data, dict)
    if data_is_dict:
        data_len = len(data)
        if not all(prop in data for prop in ['name', 'detector', 'visual']):
            raise JsonSchemaException("data must contain ['name', 'detector', 'visual'] properties", value=data, name="data", definition={'$id': '#collimator', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'detector': {'type': 'string'}, 'aperture': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, 'visual': {'$ref': '#/definitions/visual'}}, 'required': ['name', 'detector', 'visual']}, rule='required')
        data_keys = set(data.keys())
        if "name" in data_keys:
            data_keys.remove("name")
            data__name = data["name"]
            if not isinstance(data__name, (str)):
                raise JsonSchemaException("data.name must be string", value=data__name, name="data.name", definition={'type': 'string'}, rule='type')
        if "detector" in data_keys:
            data_keys.remove("detector")
            data__detector = data["detector"]
            if not isinstance(data__detector, (str)):
                raise JsonSchemaException("data.detector must be string", value=data__detector, name="data.detector", definition={'type': 'string'}, rule='type')
        if "aperture" in data_keys:
            data_keys.remove("aperture")
            data__aperture = data["aperture"]
            if not isinstance(data__aperture, (list)):
                raise JsonSchemaException("data.aperture must be array", value=data__aperture, name="data.aperture", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, rule='type')
            data__aperture_is_list = isinstance(data__aperture, list)
            if data__aperture_is_list:
                data__aperture_len = len(data__aperture)
                if data__aperture_len < 2:
                    raise JsonSchemaException("data.aperture must contain at least 2 items", value=data__aperture, name="data.aperture", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, rule='minItems')
                if data__aperture_len > 2:
                    raise JsonSchemaException("data.aperture must contain less than or equal to 2 items", value=data__aperture, name="data.aperture", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2}, rule='maxItems')
                for data__aperture_x, data__aperture_item in enumerate(data__aperture):
                    if not isinstance(data__aperture_item, (int, float)) or isinstance(data__aperture_item, bool):
                        raise JsonSchemaException(""+"data.aperture[{data__aperture_x}]".format(**locals())+" must be number", value=data__aperture_item, name=""+"data.aperture[{data__aperture_x}]".format(**locals())+"", definition={'type': 'number'}, rule='type')
        if "visual" in data_keys:
            data_keys.remove("visual")
            data__visual = data["visual"]
            validate___definitions_visual(data__visual)
    return data

def validate___definitions_detector(data):
    if not isinstance(data, (dict)):
        raise JsonSchemaException("data must be object", value=data, name="data", definition={'$id': '#detector', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'default_collimator': {'type': 'string'}, 'diffracted_beam': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'positioner': {'type': 'string'}}, 'required': ['name', 'diffracted_beam']}, rule='type')
    data_is_dict = isinstance(data, dict)
    if data_is_dict:
        data_len = len(data)
        if not all(prop in data for prop in ['name', 'diffracted_beam']):
            raise JsonSchemaException("data must contain ['name', 'diffracted_beam'] properties", value=data, name="data", definition={'$id': '#detector', 'type': 'object', 'properties': {'name': {'type': 'string'}, 'default_collimator': {'type': 'string'}, 'diffracted_beam': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'positioner': {'type': 'string'}}, 'required': ['name', 'diffracted_beam']}, rule='required')
        data_keys = set(data.keys())
        if "name" in data_keys:
            data_keys.remove("name")
            data__name = data["name"]
            if not isinstance(data__name, (str)):
                raise JsonSchemaException("data.name must be string", value=data__name, name="data.name", definition={'type': 'string'}, rule='type')
        if "default_collimator" in data_keys:
            data_keys.remove("default_collimator")
            data__defaultcollimator = data["default_collimator"]
            if not isinstance(data__defaultcollimator, (str)):
                raise JsonSchemaException("data.default_collimator must be string", value=data__defaultcollimator, name="data.default_collimator", definition={'type': 'string'}, rule='type')
        if "diffracted_beam" in data_keys:
            data_keys.remove("diffracted_beam")
            data__diffractedbeam = data["diffracted_beam"]
            if not isinstance(data__diffractedbeam, (list)):
                raise JsonSchemaException("data.diffracted_beam must be array", value=data__diffractedbeam, name="data.diffracted_beam", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='type')
            data__diffractedbeam_is_list = isinstance(data__diffractedbeam, list)
            if data__diffractedbeam_is_list:
                data__diffractedbeam_len = len(data__diffractedbeam)
                if data__diffractedbeam_len < 3:
                    raise JsonSchemaException("data.diffracted_beam must contain at least 3 items", value=data__diffractedbeam, name="data.diffracted_beam", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='minItems')
                if data__diffractedbeam_len > 3:
                    raise JsonSchemaException("data.diffracted_beam must contain less than or equal to 3 items", value=data__diffractedbeam, name="data.diffracted_beam", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='maxItems')
                for data__diffractedbeam_x, data__diffractedbeam_item in enumerate(data__diffractedbeam):
                    if not isinstance(data__diffractedbeam_item, (int, float)) or isinstance(data__diffractedbeam_item, bool):
                        raise JsonSchemaException(""+"data.diffracted_beam[{data__diffractedbeam_x}]".format(**locals())+" must be number", value=data__diffractedbeam_item, name=""+"data.diffracted_beam[{data__diffractedbeam_x}]".format(**locals())+"", definition={'type': 'number'}, rule='type')
        if "positioner" in data_keys:
            data_keys.remove("positioner")
            data__positioner = data["positioner"]
            if not isinstance(data__positioner, (str)):
                raise JsonSchemaException("data.positioner must be string", value=data__positioner, name="data.positioner", definition={'type': 'string'}, rule='type')
    return data

def validate___definitions_visual(data):
    if not isinstance(data, (dict)):
        raise JsonSchemaException("data must be object", value=data, name="data", definition={'$id': '#visual', 'type': 'object', 'properties': {'pose': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, 'colour': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'mesh': {'type': 'string'}}, 'required': ['mesh']}, rule='type')
    data_is_dict = isinstance(data, dict)
    if data_is_dict:
        data_len = len(data)
        if not all(prop in data for prop in ['mesh']):
            raise JsonSchemaException("data must contain ['mesh'] properties", value=data, name="data", definition={'$id': '#visual', 'type': 'object', 'properties': {'pose': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, 'colour': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, 'mesh': {'type': 'string'}}, 'required': ['mesh']}, rule='required')
        data_keys = set(data.keys())
        if "pose" in data_keys:
            data_keys.remove("pose")
            data__pose = data["pose"]
            if not isinstance(data__pose, (list)):
                raise JsonSchemaException("data.pose must be array", value=data__pose, name="data.pose", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, rule='type')
            data__pose_is_list = isinstance(data__pose, list)
            if data__pose_is_list:
                data__pose_len = len(data__pose)
                if data__pose_len < 6:
                    raise JsonSchemaException("data.pose must contain at least 6 items", value=data__pose, name="data.pose", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, rule='minItems')
                if data__pose_len > 6:
                    raise JsonSchemaException("data.pose must contain less than or equal to 6 items", value=data__pose, name="data.pose", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6}, rule='maxItems')
                for data__pose_x, data__pose_item in enumerate(data__pose):
                    if not isinstance(data__pose_item, (int, float)) or isinstance(data__pose_item, bool):
                        raise JsonSchemaException(""+"data.pose[{data__pose_x}]".format(**locals())+" must be number", value=data__pose_item, name=""+"data.pose[{data__pose_x}]".format(**locals())+"", definition={'type': 'number'}, rule='type')
        if "colour" in data_keys:
            data_keys.remove("colour")
            data__colour = data["colour"]
            if not isinstance(data__colour, (list)):
                raise JsonSchemaException("data.colour must be array", value=data__colour, name="data.colour", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='type')
            data__colour_is_list = isinstance(data__colour, list)
            if data__colour_is_list:
                data__colour_len = len(data__colour)
                if data__colour_len < 3:
                    raise JsonSchemaException("data.colour must contain at least 3 items", value=data__colour, name="data.colour", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='minItems')
                if data__colour_len > 3:
                    raise JsonSchemaException("data.colour must contain less than or equal to 3 items", value=data__colour, name="data.colour", definition={'type': 'array', 'items': {'type': 'number'}, 'minItems': 3, 'maxItems': 3}, rule='maxItems')
                for data__colour_x, data__colour_item in enumerate(data__colour):
                    if not isinstance(data__colour_item, (int, float)) or isinstance(data__colour_item, bool):
                        raise JsonSchemaException(""+"data.colour[{data__colour_x}]".format(**locals())+" must be number", value=data__colour_item, name=""+"data.colour[{data__colour_x}]".format(**locals())+"", definition={'type': 'number'}, rule='type')
        if "mesh" in data_keys:
            data_keys.remove("mesh")
            data__mesh = data["mesh"]
            if not isinstance(data__mesh, (str)):
                raise JsonSchemaException("data.mesh must be string", value=data__mesh, name="data.mesh", definition={'type': 'string'}, rule='type')
    return data