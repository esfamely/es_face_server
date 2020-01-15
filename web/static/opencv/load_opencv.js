//---------------
//异步加载opencv相关组件
//---------------

// 人脸检测器
let classifier = null;

// 加载haarcascade人脸检测器
function onOpenCvReady1(url) {
	let path = 'haarcascade_frontalface_default.xml';
	let request = new XMLHttpRequest();
	request.open('GET', url, true);
	request.responseType = 'arraybuffer';
	request.onload = function(ev) {
		if (request.readyState === 4) {
			if (request.status === 200) {
				let data = new Uint8Array(request.response);
				cv.FS_createDataFile('/', path, data, true, false, false);
				onOpenCvReady2();
			} else {
				alert('人脸检测器加载失败：' + request.status);
			}
		}
	};
	request.send();
}

function onOpenCvReady2() {
	classifier = new cv.CascadeClassifier();

	// load pre-trained classifiers
	classifier.load('haarcascade_frontalface_default.xml');

	$('#status').hide();
	onOpenCvReady();
}
