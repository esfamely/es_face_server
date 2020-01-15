//---------------
//人脸采集器
//---------------

// 开启模式
let processMode = 1;
// 采集模式，-1=强制停止采集，0=不采集，1=采集，2=识别并采集，3=连续采集
let collectMode = 0;
// 人员id
let cUid = null;
// 人脸图像缓存
let faceCaches = [];
// 帧计数
let index = 0;
let indexMax = 100000;
// 当前时间
let dt = null;
// 视频处理间隔
let interval = 10;
// 每隔几帧进行一次人脸检测
let frameFD = 5;
// 上一次检测到的人脸区域
let lastFaceRect = null;

// 初始化缓存信息
function clearCache() {
	// 释放图像资源
	for (let i=0; i<faceCaches.length; i++) {
		faceCaches[i].delete();
	}
	faceCaches = [];
}

// 进入或切换到某个人员的采集模式
function toCollect(uid) {
	processMode = 0;
	clearCache();
	cUid = uid;
	collectMode = 1;
	processMode = 1;
}

// 进入或切换到识别模式
function toPredict() {
	processMode = 0;
	clearCache();
	collectMode = 2;
	processMode = 1;
}

// 进入或切换到连续采集模式
function toCollectX(to) {
	processMode = 0;
	if (to != -1) {
		clearCache();
	}
	collectMode = to;
	processMode = 1;
}

function setupProcess() {
	// 视频帧
	let src = new cv.Mat(videoUI.height, videoUI.width, cv.CV_8UC4);
	let cap = new cv.VideoCapture(videoUI);
	// 缩小图像加快检测速度
	let imgMini = new cv.Mat();
	// 检测到的人脸区域
	let faces = new cv.RectVector();
	return {
		src: src,
		cap: cap,
		imgMini: imgMini,
		faces: faces
	};
}

function startProcess(src, cap, imgMini, faces) {
	cap.read(src);
	let img = process(src, index, imgMini, faces);

	// 计算FPS
	calFPS(index);

	playImg(img);
	index = (index + 1) % indexMax;
	setTimeout(() => {
		startProcess(src, cap, imgMini, faces);
	}, interval);
}

function process(img, i, imgMini, faces) {
	if (processMode == 0) {
		return img;
	}

	let width = videoUI.width;
	let height = videoUI.height;

	// 缩放因子
	let sFactorMini = setup['s1_factor_mini'];
	// 人脸边界预留宽度
	let sBorder = setup['s1_border'];
	// 人脸图像统一尺寸
	let sSize = setup['s1_size'];
	// 每隔几帧保存一次人脸图像
	let sFrameCc = setup['s1_frame_cc'];

	let face = null;
	// 为了提高fps，并非每帧都进行人脸检测，不进行检测的帧取上一次的检测结果
	if (i % frameFD == 0) {
		// 缩小图像加快检测速度
		cv.resize(img, imgMini, {width: Math.floor(width * sFactorMini), height: Math.floor(height * sFactorMini)});

		// 人脸检测
		classifier.detectMultiScale(imgMini, faces, 1.1, 3, 0);
		// 只要最大的图
		face = max_rect(faces);
		lastFaceRect = face;
	} else {
		face = lastFaceRect;
	}
	if (face == null) {
		return img;
	}

	// 人脸标记框左上点与右下点
	let point1 = new cv.Point(Math.ceil(face.x / sFactorMini) - sBorder, Math.ceil(face.y / sFactorMini) - sBorder);
	let point2 = new cv.Point(Math.ceil((face.x + face.width) / sFactorMini) + sBorder, Math.ceil((face.y + face.height) / sFactorMini) + sBorder);

	// 越界的图不要
	if (point1.x <= 0 || point1.y <= 0 || point2.x >= width || point2.y >= height) {
		return img;
	}

	// 标记人脸区域
	cv.rectangle(img, point1, point2, [255, 255, 255, 255], 5);

	// 采集模式下，每隔几帧保存一次人脸图像
	if (collectMode >= 1 && i % sFrameCc == 0) {
		// 提取人脸图像
		let imgFace = new cv.Mat();
		imgFace = img.roi(new cv.Rect(point1.x, point1.y, point2.x - point1.x, point2.y - point1.y));
		// 保存
		saveToCache(imgFace);
	}

	return img;
}

function calFPS(i) {
	if (i % 10 == 0) {
		if (dt == null) {
			dt = Date.now();
		} else {
			let fps = Math.ceil(10 * 1000 / (Date.now() - dt));
			dt = Date.now();
			parent.window.setFPS(fps);
		}
	}
}

function playImg(img) {
	// 照镜效果
	cv.flip(img, img, 1);

	// 显示帧图像
	cv.imshow(canvasUI, img);
}

function saveToCache(imgFace) {
	// 保存满几张就提交
	let sSubmitCc = setup['s1_submit_cc'];

	faceCaches.push(imgFace);
	cv.imshow('tmpCanvas' + faceCaches.length, imgFace);

	// 保存达到一定数量就提交
	if (faceCaches.length >= sSubmitCc) {
		// 提交前停止采集
		let collectModeOld = collectMode;
		collectMode = 0;

		if (3 == collectModeOld) {
			// 连续采集
			submitFaceX();
		} else {
			// 提交
			submitFace(collectModeOld);
		}
	}
}

function submitFace(collectModeOld) {
	var formData = new FormData();
	formData.append('uid', cUid);
	formData.append('img_cc', faceCaches.length);
	formData.append('to_rec', 2 == collectModeOld ? 1 : 0);

	// 转为blob并上传
	imgToBlob(formData, 0, () => {
		fetch(serverUrl + '/collection/collect', {
			method: 'POST',
			body: formData
		})
		.then(res => res.json())
		.then(
			(result) => {
				//alert(JSON.stringify(result));

				// 完成采集
				onSubmitOk(result);
			},
			(error) => {
				alert('上传图像出现异常：' + error);
			}
		);
	});
}

function submitFaceX() {
	var formData = new FormData();
	formData.append('img_cc', faceCaches.length);

	// 转为blob并上传
	imgToBlob(formData, 0, () => {
		fetch(serverUrl + '/collection/collectX', {
			method: 'POST',
			body: formData
		})
		.then(res => res.json())
		.then(
			(result) => {
				// 完成采集
				onSubmitOk(result);
				// 继续连续采集
				collectMode = (-1 == collectMode ? 0 : 3);
			},
			(error) => {
				//alert('上传图像出现异常：' + error);
			}
		);
	});
}

function imgToBlob(formData, i, callback) {
	if (i >= faceCaches.length) {
		if (callback) {
			callback();
		}
	}

	let canvas = document.querySelector('#tmpCanvas' + (i + 1));
	canvas.toBlob(function (blob) {
		formData.append('file' + i, blob);

		imgToBlob(formData, ++i, callback);
	}, 'image/jpg');
}

function onSubmitOk(result) {
	if (result) {
		parent.window.setCollectResult(result);
	}
	// 初始化缓存信息
	clearCache();
}
