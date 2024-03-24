from flask_restful import Resource, reqparse
from modelEvaluator import MlModel


class Evaluator(Resource):

    def post(self):
        parser = reqparse.RequestParser()
        # validate the input, if not valid, return 401
        parser.add_argument("sampleModelAddress", type=str)
        parser.add_argument("testDataAddress", type=str)
        parser.add_argument("testLabelAddress", type=str)
        parser.add_argument("clientWeights", action='append')
        args = parser.parse_args()
        if not args['clientWeights']:
            return {'error': 'clientWeights is required'}, 401
        if not args['testDataAddress']:
            return {'error': 'testDataAddress is required'}, 401
        if not args['testLabelAddress']:
            return {'error': 'testLabelAddress is required'}, 401
        # if not args['sampleModelAddress']:
        #     return {'error': 'sampleModelAddress is required'}, 401
        model = MlModel(args['clientWeights'], args['testDataAddress'], args['testLabelAddress']) #TODO: Add model address
        results = model.get_evaluation_results()
        return {'results': results}
    
    def get(self):
        return {'data': 'get'}



