from flask_restful import Resource, reqparse
from app.modelEvaluator import MlModel


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
        new_model = "QmXvmaD8FuPnySgNaxv3vun9ZtuMGdDFnNS6tsLKz8Jhyj"
        return {'results': results, 'new_model': new_model}, 200
    
    def get(self):
        return {'message': 'The model evaluator is running!'}, 200



