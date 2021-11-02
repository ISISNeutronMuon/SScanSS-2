log_config = {
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {"format": "%(asctime)s - %(threadName)s -  " "%(name)s - %(levelname)s - " "%(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "backupCount": 10,
            "class": "logging.handlers.RotatingFileHandler",
            "encoding": "utf8",
            "filename": "info.log",
            "formatter": "verbose",
            "level": "INFO",
            "maxBytes": 10485760,
        },
    },
    "loggers": {"my_module": {"handlers": ["console"], "level": "DEBUG", "propagate": "no"}},
    "root": {"handlers": ["console", "file_handler"], "level": "INFO"},
    "version": 1,
}

schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "definitions": {
        "collimator": {
            "$id": "#collimator",
            "properties": {
                "aperture": {"items": {"type": "number"}, "maxItems": 2, "minItems": 2, "type": "array"},
                "detector": {"type": "string"},
                "name": {"type": "string"},
                "visual": {"$ref": "#/definitions/visual"},
            },
            "required": ["name", "detector", "aperture", "visual"],
            "type": "object",
        },
        "detector": {
            "$id": "#detector",
            "properties": {
                "default_collimator": {"type": "string"},
                "diffracted_beam": {"items": {"type": "number"}, "maxItems": 3, "minItems": 3, "type": "array"},
                "name": {"type": "string"},
                "positioner": {"type": "string"},
            },
            "required": ["name", "diffracted_beam"],
            "type": "object",
        },
        "hardware": {
            "$id": "#hardware",
            "properties": {"name": {"type": "string"}, "visual": {"$ref": "#/definitions/visual"}},
            "required": ["name", "visual"],
            "type": "object",
        },
        "joint": {
            "$id": "#joint",
            "properties": {
                "axis": {"items": {"type": "number"}, "maxItems": 3, "minItems": 3, "type": "array"},
                "child": {"type": "string"},
                "home_offset": {"type": "number"},
                "lower_limit": {"type": "number"},
                "name": {"type": "string"},
                "origin": {"items": {"type": "number"}, "maxItems": 3, "minItems": 3, "type": "array"},
                "parent": {"type": "string"},
                "type": {"enum": ["prismatic", "revolute"], "type": "string"},
                "upper_limit": {"type": "number"},
            },
            "required": ["name", "type", "parent", "child", "axis", "origin", "lower_limit", "upper_limit"],
            "type": "object",
        },
        "link": {
            "$id": "#link",
            "properties": {"name": {"type": "string"}, "visual": {"$ref": "#/definitions/visual"}},
            "required": ["name"],
            "type": "object",
        },
        "positioner": {
            "$id": "#positioner",
            "properties": {
                "base": {"items": {"type": "number"}, "maxItems": 6, "minItems": 6, "type": "array"},
                "custom_order": {"items": {"type": "string"}, "type": "array"},
                "joints": {"items": {"$ref": "#/definitions/joint"}, "minItems": 1, "type": "array"},
                "links": {"items": {"$ref": "#/definitions/link"}, "minItems": 2, "type": "array"},
                "name": {"type": "string"},
                "tool": {"items": {"type": "number"}, "maxItems": 6, "minItems": 6, "type": "array"},
            },
            "required": ["name", "joints", "links"],
            "type": "object",
        },
        "positioning_stack": {
            "$id": "#positioning_stack",
            "properties": {
                "name": {"type": "string"},
                "positioners": {"items": {"type": "string"}, "minItems": 1, "type": "array"},
            },
            "required": ["name", "positioners"],
            "type": "object",
        },
        "visual": {
            "$id": "#visual",
            "properties": {
                "colour": {"items": {"type": "number"}, "maxItems": 3, "minItems": 3, "type": "array"},
                "mesh": {"type": "string"},
                "pose": {"items": {"type": "number"}, "maxItems": 6, "minItems": 6, "type": "array"},
            },
            "required": ["mesh"],
            "type": "object",
        },
    },
    "description": "SScanSS 2 instrument",
    "properties": {
        "instrument": {
            "properties": {
                "collimators": {"items": {"$ref": "#/definitions/collimator"}, "type": "array"},
                "detectors": {"items": {"$ref": "#/definitions/detector"}, "minItems": 1, "type": "array"},
                "fixed_hardware": {"items": {"$ref": "#/definitions/hardware"}, "type": "array"},
                "gauge_volume": {
                    "description": "The " "centre " "of " "gauge " "volume",
                    "items": {"type": "number"},
                    "maxItems": 3,
                    "minItems": 3,
                    "type": "array",
                },
                "incident_jaws": {
                    "properties": {
                        "aperture": {"items": {"type": "number"}, "maxItems": 2, "minItems": 2, "type": "array"},
                        "aperture_lower_limit": {
                            "items": {"type": "number"},
                            "maxItems": 2,
                            "minItems": 2,
                            "type": "array",
                        },
                        "aperture_upper_limit": {
                            "items": {"type": "number"},
                            "maxItems": 2,
                            "minItems": 2,
                            "type": "array",
                        },
                        "beam_direction": {"items": {"type": "number"}, "maxItems": 3, "minItems": 3, "type": "array"},
                        "beam_source": {"items": {"type": "number"}, "maxItems": 3, "minItems": 3, "type": "array"},
                        "positioner": {"type": "string"},
                        "visual": {"$ref": "#/definitions/visual"},
                    },
                    "required": [
                        "aperture",
                        "aperture_lower_limit",
                        "aperture_upper_limit",
                        "beam_direction",
                        "beam_source",
                        "visual",
                    ],
                    "type": "object",
                },
                "name": {"description": "The " "unique " "name of " "the " "instrument", "type": "string"},
                "positioners": {"items": {"$ref": "#/definitions/positioner"}, "minItems": 1, "type": "array"},
                "positioning_stacks": {
                    "items": {"$ref": "#/definitions/positioning_stack"},
                    "minItems": 1,
                    "type": "array",
                },
                "script_template": {"description": "The " "path " "of " "script " "template", "type": "string"},
                "version": {"description": "The " "version " "number " "of " "file", "type": "string"},
            },
            "required": [
                "name",
                "version",
                "gauge_volume",
                "incident_jaws",
                "detectors",
                "positioning_stacks",
                "positioners",
            ],
            "type": "object",
        }
    },
    "required": ["instrument"],
    "title": "Instrument",
    "type": "object",
}
