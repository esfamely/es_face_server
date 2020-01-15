let name = 'audience';
let toName = 'movie_' + cameraSn;
let connection = new WebSocket('wss://192.168.1.36:6001');

let videoTips = document.querySelector('#video-tips');
let theirVideo = document.querySelector('#theirs');
let conns = {};
let stream;

videoTips.style.display = 'block';
theirVideo.style.display = 'none';

// 连接并登录 Signaling 服务器
connection.onopen = () => {
	console.log('connected');

	send({
		type: 'login'
	}, name);
};

// Handle all messages through this callback
connection.onmessage = message => {
	console.log('got message', message.data);

	var data = JSON.parse(message.data);
	switch(data.type) {
		case 'login':
			onLogin(data.success, data.from, data.to);
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

connection.onerror = err => {
	console.log('got error', err);
};

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

function onLogin(success, from, to) {
	if (success === false) {
		alert('当前观看人数已满，请稍候再看！');
	} else {
		// Get the plumbing ready for a call
		startConnection(to, toName);
	}
}

function startConnection(from, to) {
	if (hasRTCPeerConnection()) {
		startPeerConnection(from, to);
	} else {
		alert('sorry, your browser does not support WebRTC.');
	}
}

function setupPeerConnection(yourConnection, from, to) {
	// Setup stream listening
	yourConnection.onaddstream = e => {
		videoTips.style.display = 'none';
		theirVideo.style.display = 'block';

		stream = e.stream;
		try {
			theirVideo.srcObject = stream;
		} catch (error) {
			theirVideo.src = window.URL.createObjectURL(stream);
		}
	};

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

function startPeerConnection(from, to) {
	yourConnection = getConnection(from, to);
	setupPeerConnection(yourConnection, from, to);
	
	// Begin the offer
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

function onOffer(offer, from, to) {
	yourConnection = getConnection(from, to);
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
	theirVideo.src = null;
	yourConnection.close();
	yourConnection.onicecandidate = null;
	yourConnection.onaddstream = null;
	delConnection(from, to);
};
