from ...main import api
from celery.result import AsyncResult
from flask_restplus import Resource, fields


parser = api.parser()
parser.add_argument('task_id', location='values',
                    help='Task ID', required=True)

model = api.model(
    'Task Status', {'state': fields.String, 'result': fields.String})


@api.route('/status/<task_id>')
@api.expect(parser)
class Status(Resource):
    @api.marshal_with(model)
    def get(self, task_id):
        task = AsyncResult(task_id)
        if task.state == 'PENDING':
            # job did not start yet
            response = {
                'state': task.state,
                'status': 'Pending...'
            }
        else:
            response = {
                'state': task.state,
                'result': task.result
            }
        return response
