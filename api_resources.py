from flask_restful import Resource

import model
import parse_request


class Text(Resource):
    path = '/text'

    def put(self):
        args = parse_request.request_put_text()
        try:
            prediction = model.predict_text(args['text'])
            print(prediction)
        except Exception as e:
            raise Exception("ERROR IN PREDICTION -- " + str(e))
