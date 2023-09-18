from flask import Blueprint
from flask import jsonify, request, send_from_directory


import apps.service.few_shot_test as few_shot_test
import apps.service.image_decode as img_dc
import apps.service.get_url as gurl

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

bp = Blueprint(name='bp',
                       import_name=__name__,
                       url_prefix='/fsl')

@bp.route('/test', methods=['POST'])
def get_test_result():
    json_data = request.get_json()

    img = img_dc.image_decoding(json_data) # 이미지 디코딩
    result = few_shot_test.find_class() # 쿼리 이미지 넣어서 테스트 돌리기
    purl = gurl.get_product_url(result)

    return purl

# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @bp.route('/images/<path:path>')
# def send_image(path):
#     if allowed_file(path):
#         print(path)
#         return send_from_directory('static/images', path)
#     else:
#         return 'Invalid file type'