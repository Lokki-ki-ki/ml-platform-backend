from flask_restful import Resource, reqparse
from app.modelEvaluator import MlModel


class Evaluator(Resource):

    def post(self):
        parser = reqparse.RequestParser()
        # validate the input, if not valid, return 401
        # Example request body:
        # {
        #     "clientsToSubmissions":{"1":"QmXvmaD8FuPnySgNaxv3vun9ZtuMGdDFnNS6tsLKz8Jhyj","2":"QmXvmaD8FuPnySgNaxv3vun9ZtuMGdDFnNS6tsLKz8Jhyj","3":"QmXvmaD8FuPnySgNaxv3vun9ZtuMGdDFnNS6tsLKz8Jhyj"},
        #     "clientsToReputation":{"1":"100","2":"100","3":"100"},
        #     "rewardPool":"999999999999999900",
        #     "modelAddress":"0x1234567890123456789012345678901234567890",
        #     "testDataAddress":"0x1234567890123456789012345678901234567890",
        #     "testLabelAddress":"0x1234567890123456789012345678901234567890",
        #     "testDataHash":"0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
        #     "testLabelHash":"0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
        # }
        parser.add_argument("clientsToSubmissions", type=dict)
        parser.add_argument("clientsToReputation", type=dict)
        parser.add_argument("rewardPool", type=str)
        parser.add_argument("modelAddress", type=str)
        parser.add_argument("testDataAddress", type=str)
        parser.add_argument("testLabelAddress", type=str)
        parser.add_argument("testDataHash", type=str)
        parser.add_argument("testLabelHash", type=str)
        args = parser.parse_args()
        if not args['clientsToSubmissions']:
            return {'error': 'clientsToSubmissions is required'}, 401
        if not args['clientsToReputation']:
            return {'error': 'clientsToReputation is required'}, 401
        if not args['rewardPool']:
            return {'error': 'rewardPool is required'}, 401
        if not args['modelAddress']:
            return {'error': 'modelAddress is required'}, 401
        if not args['testDataAddress']:
            return {'error': 'testDataAddress is required'}, 401
        if not args['testLabelAddress']:
            return {'error': 'testLabelAddress is required'}, 401
        
        # def __init__(self, client_weights, test_data_add=TEST_DATA, test_labels_add=TEST_LABELS, model_address=None, reward_pool=None, clientsToReputation=None, clientsToSubmissions=None, test_data_hash=None, test_label_hash=None):
        model = MlModel(args['clientsToSubmissions'], args['testDataAddress'], args['testLabelAddress'], args['modelAddress'], args['rewardPool'], args['clientsToReputation'], args['testDataHash'], args['testLabelHash']) #TODO: Add model address
        results = model.get_evaluation_results()
        # Results shoule contains:
        # const newModelAddress = "0x1234567890123456789012345678901234567890";
        # const clientIds = [1, 2, 3];
        # const clientNewReputations = [100, 100, 100];
        # const clientRewards = [80, 10, 10];
        return results, 200
    
    def get(self):
        return {'message': 'The model evaluator is running!'}, 200



