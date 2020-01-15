// 服务器url
let serverUrl = 'https://192.168.1.141:7036/face';
// 系统配置
let setup = {};
// 是否使用视频远程监控服务
let useRVS = 0;
// 视频播放器
let videoUI = null;
// 视频显示器
let canvasUI = null;

function init() {
	loadSetup(() => {
		onOpenCvReady1(setup['s1_cascade_path']);
	});

	videoUI = document.querySelector('#videoUI');
	canvasUI = document.querySelector('#canvasUI');
}

// 读取系统配置
function loadSetup(callback) {
	fetch(serverUrl + '/load_setup', {
		method: 'POST'
	})
	.then(res => res.json())
	.then(
		(result) => {
			setup = result;
			//alert(JSON.stringify(setup));
			if (callback) {
				callback();
			}
		},
		(error) => {
			alert('读取系统配置异常：' + error);
		}
	);
}

function onOpenCvReady() {
	// 居中显示
	canvasUI.style.display = 'block';
	canvasUI.style.paddingLeft = (document.body.clientWidth - videoUI.width) / 2 + 'px';

	if (1 == useRVS) {
		// 启动视频远程监控服务...
		let onLoginOk = () => {
			startPlay(videoUI, () => {
				// 并分析播放视频
				r = setupProcess();
				startProcess(r.src, r.cap, r.imgMini, r.faces);
			});
		};
		setupConnection(setup['camera_sn'], onLoginOk);
	} else {
		navigator.mediaDevices.getUserMedia({ video: true, audio: false }).then(myStream => {
			videoUI.srcObject = myStream;

			// 分析播放视频
			r = setupProcess();
			startProcess(r.src, r.cap, r.imgMini, r.faces);
		}).catch(function(e) {
			alert('视频播放异常：' + e);
		});
	}
}
