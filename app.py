from flask import Flask
from flask_restful import Api
import flask_restful
import api_resources


class App:

    def __init__(self, name, config=""):
        self.app = Flask(name)
        self.api = Api(self.app)
        if config:
            self._config(config)
        self._add_resources()

    def _config(self, config):
        # TODO: custom app configurations if need be
        pass

    def _add_resources(self):
        """
        Add all Resource derived classes from the api_resources module
        to the api
        """
        for key, val in api_resources.__dict__.items():
            if hasattr(val, '__bases__'):
                base_class = val.__bases__
                if base_class[0] == flask_restful.Resource:
                    try:
                        self.api.add_resource(val, val.path)
                    except Exception as e:
                        raise Exception("ERROR IN ADDING RESOURCES -- " + str(e))


if __name__ == "__main__":
    app_ADR = App(__name__)
    app_ADR.app.run(debug=True)
