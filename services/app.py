from ipaddress import ip_address
import sys
sys.path.append('.')
sys.path.append('..')

import argparse
from flask import Flask, render_template, request
import os
import yaml 
import urllib
from configs.config import Config
from demo.demo import Demo 
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/src'
app.config['RESULT_FOLDER'] = 'static/results'
ip_address = 'http://selab.nhtlongcs.com:20690/'
 
@app.route('/',  methods=['GET', 'POST'])
def refer_and_render():
    image_path = request.args.get("image_path", default = "test.jpg", type=str)
    expression = request.args.get("expression", default = "*", type = str) 
    if image_path.startswith("http"):
        new_image_path = str(uuid.uuid4()) + ".jpg"
        urllib.request.urlretrieve(image_path, os.path.join(app.config["UPLOAD_FOLDER"], new_image_path))
        image_path = new_image_path
    full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path)
    if expression == "*":
        return render_template("index.html", user_image = full_image_path)
    else:
        out_path = demo.refer(full_image_path, expression, app.config['RESULT_FOLDER'])
        return render_template("refer.html", text = expression, user_image = out_path)


@app.route('/refer',  methods=['GET', 'POST'])
def refer():
    image_path = request.args.get("image_path", default = "test.jpg", type=str)
    expression = request.args.get("expression", default = "*", type = str) 
    if image_path.startswith("http"):
        new_image_path = str(uuid.uuid4()) + ".jpg"
        urllib.request.urlretrieve(image_path, os.path.join(app.config["UPLOAD_FOLDER"], new_image_path))
        image_path = new_image_path
    full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_path)
    if expression == "*":
        return "No expression provided."
    else:
        out_path = demo.refer(full_image_path, expression, app.config['RESULT_FOLDER'])
        return os.path.join(ip_address, out_path)
        # return render_template("refer.html", text = expression, user_image = out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml', help='Configuration')
    args = parser.parse_args()
    config = Config(yaml.load(open(args.config, 'r'), Loader=yaml.Loader))

    demo = Demo(config.APP.MODEL.CONFIG, config.APP.MODEL.WEIGHT)
    port = int(os.environ.get('PORT', config.APP.PORT))
    app.run(host="0.0.0.0", port = port, debug = True, use_reloader = False)

