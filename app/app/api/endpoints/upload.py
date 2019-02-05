from ...main import api
from flask_restplus import Resource, fields
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
import zipfile
import time
import os

from . import utils

upload_parser = api.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True, help='zip archive or single file to upload')

model = api.model(
    'Status', {'status': fields.String})


@api.route('/upload')
@api.expect(upload_parser)
class Upload(Resource):
    @api.marshal_with(model, code=201)
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['file']  # This is FileStorage instance
        filename = secure_filename(uploaded_file.filename)
        filepath = utils.data_dir + 'upload/' + time.strftime("%Y%m%d-%H%M%S") + filename
        uploaded_file.save(filepath)
        if filename.rsplit('.', 1)[1].lower() in ['zip']:
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                zip_ref.extractall(utils.data_dir + '/raw/')
        else:
            os.rename(filepath, utils.data_dir + '/raw/' + filename)
        utils.gitCommitData(filename)
        return {'status': 'OK'}
