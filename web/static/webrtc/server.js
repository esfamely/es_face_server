let name = 'movie';
// Signaling服务器url
let webSocketUrl = 'wss://192.168.1.36:6001';
let connection = null;
let conns = {};
let stream;
let onLoginOk = null;

function setupConnection(cameraSn, onLoginOk) {
	// 连接Signaling服务器
	try {
		connection = new WebSocket(webSocketUrl);
	} catch (e) {
		alert('视频信号服务器连接失败：' + e);
	}

	// 登录
	connection.onopen = () => {
		console.log('connected');

		send({
			type: 'login'
		}, name + '_' + cameraSn);
	};

	// Handle all messages through this callback
	connection.onmessage = message => {
		console.log('got message', message.data);

		var data = JSON.parse(message.data);
		switch(data.type) {
			case 'login':
				onLogin(data.success, data.from, data.to, onLoginOk);
				break;
			case 'offer':
				onOffer(data.offer, data.from, data.to);
				break;
			case 'answer':
				onAnswer(data.answer, data.from, data.to);
				break;
			case 'candidate':
				onCandidate(data.candidate, data.from, data.to);
				break;
			case 'leave':
				onLeave(data.from, data.to);
				break;
			default:
				break;
		}
	};

	connection.onerror = e => {
		alert('视频信号服务器连接失败：' + JSON.stringify(e));
	};
}

// Alias for sending messages in JSON format
function send(message, from, to) {
	// 确保发送方是自己，否则交换
	if (from.indexOf(name) < 0) {
		let tmp = from;
		from = to;
		to = tmp;
	}

	message.from = from;
	if (to) {
		message.to = to;
	}

	connection.send(JSON.stringify(message));
}

function onLogin(success, from, to, onLoginOk) {
	if (success === false) {
		alert('你不是合法的院线，无法登录！');
	} else {
		// Get the plumbing ready for a call
		//startPlay();
		if (onLoginOk) {
			onLoginOk();
		}
	}
}

function startPlay(videoUI, callback) {
	if (hasUserMedia()) {
		navigator.mediaDevices.getUserMedia({ video: true, audio: false }).then(myStream => {
			stream = myStream;
			//videoUI.style.display = 'block';
			try {
				videoUI.srcObject = stream;
			} catch (error) {
				videoUI.src = window.URL.createObjectURL(stream);
			}

			if (hasRTCPeerConnection()) {
				// 取消即时连接，有观众呼叫时才建立连接
				//setupPeerConnection();
			} else {
				alert('sorry, your browser does not support WebRTC.');
			}

			if (callback) {
				callback();
			}
		}).catch(function(e) {
			alert('视频播放异常：' + e);
		});
	} else {
		alert('sorry, your browser does not support WebRTC.');
	}
}

function setupPeerConnection(yourConnection, from, to) {
	// Setup stream listening
	yourConnection.addStream(stream);

	// Setup ice handling
	yourConnection.onicecandidate = event => {
		if (event.candidate) {
			send({
				type: 'candidate',
				candidate: event.candidate
			}, from, to);
		}
	};
}

function onOffer(offer, from, to) {
	yourConnection = getConnection(from, to);
	setupPeerConnection(yourConnection, from, to);

	yourConnection.setRemoteDescription(new RTCSessionDescription(offer));
	yourConnection.createAnswer().then(answer => {
		yourConnection.setLocalDescription(answer);
		send({
			type: "answer",
			answer: answer
		}, from, to);
	}, function (error) {
		alert("an error has occurred");
	});

	// 服务器收到客户端发信号后，需要再向客户端发信号，才可以建立媒体流通讯链接
	yourConnection.createOffer().then(offer => {
		send({
			type: "offer",
			offer: offer
		}, from, to);
		yourConnection.setLocalDescription(offer);
	}, function (error) {
		alert("an error has occurred.");
	});
}

function onAnswer(answer, from, to) {
	yourConnection = getConnection(from, to);
	yourConnection.setRemoteDescription(new RTCSessionDescription(answer));
}

function onCandidate(candidate, from, to) {
	yourConnection = getConnection(from, to);
	yourConnection.addIceCandidate(new RTCIceCandidate(candidate));
}

function onLeave(from, to) {
	yourConnection = getConnection(from, to);
	yourConnection.close();
	yourConnection.onicecandidate = null;
	delConnection(from, to);
};
