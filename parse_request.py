from flask_restful import reqparse


def request_put_text():
    put_args = reqparse.RequestParser()
    put_args.add_argument("text", type=str, help="Some text is required", required=True)
    return put_args.parse_args()
