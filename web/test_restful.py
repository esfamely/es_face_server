from flask import Flask, abort, request, render_template, jsonify
import json
from time import sleep

app = Flask(__name__, template_folder='templates', static_folder='static')

# 测试数据暂时存放
tasks = []


@app.route('/', methods=['GET'])
def index():
    return render_template("index_img_.html")


# test: http://127.0.0.1:3600/add_task?id=123&info=ko
@app.route('/add_task', methods=['GET'])
def add_task():
    if not request.args or 'id' not in request.args or 'info' not in request.args:
        abort(400)
    task = {
        'id': request.args['id'],
        'info': request.args['info']
    }
    tasks.append(task)
    return json.dumps({'result': 'success'})


# test: http://127.0.0.1:3600/get_task?id=123
@app.route('/get_task', methods=['GET'])
def get_task():
    if not request.args or 'id' not in request.args:
        # 没有指定id则返回全部
        return json.dumps(tasks)
    else:
        task_id = request.args['id']
        #task = filter(lambda t: t['id'] == int(task_id), tasks)
        #return jsonify(task) if task else jsonify({'result': 'not found'})

        for task in tasks:
            #print("{} | {}".format(task['id'], int(task_id)))
            if task['id'] == task_id:
                #print("ko")
                return json.dumps(task)
        return json.dumps({'result': 'not found'})


@app.route('/test1/<ko>', methods=["get", "POST"])
def test1(ko):
    print(ko)
    #sleep(3)
    print(request.headers["User-Agent"])
    return json.dumps({'result': '123'})


@app.route('/send_img', methods=['GET', 'POST'])
def send_img():
    '''if request.method == 'POST':
        f1 = request.files['file1']
        f2 = request.files['file2']
        print(request.form['info'])
        f1.save('D:/{}'.format(f1.filename))
        f2.save('D:/{}'.format(f2.filename))'''
    print(request.form["tel"])
    file = request.files["fs"]
    print(file)
    file.save("D:/123.jpg")
    return json.dumps({'result': 'success'})


if __name__ == "__main__":
    # 将host设置为0.0.0.0，则外网用户也可以访问到这个服务
    app.run(host="127.0.0.1", port=3600, debug=True)
