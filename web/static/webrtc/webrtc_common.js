function hasUserMedia() {
	navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
	return !!navigator.getUserMedia;
}

function hasRTCPeerConnection() {
	window.RTCPeerConnection = window.RTCPeerConnection || window.webkitRTCPeerConnection || window.mozRTCPeerConnection;
	window.RTCSessionDescription = window.RTCSessionDescription || window.webkitRTCSessionDescription || window.mozRTCSessionDescription;
	window.RTCIceCandidate = window.RTCIceCandidate || window.webkitRTCIceCandidate || window.mozRTCIceCandidate;
	return !!window.RTCPeerConnection;
}

// 通过收发双方名获取p2p连接
function getConnection(from, to) {
	// 先查找连接池...
	let testName = from + '_' + to;
	if (conns[testName]) {
		return conns[testName];
	}
	testName = to + '_' + from;
	if (conns[testName]) {
		return conns[testName];
	}

	// 无则新建连接
	var configuration = {
		//'iceServers': [{ 'url': 'stun:stun.1.google.com:19302' }]
	};
	let yourConnection = new RTCPeerConnection(configuration);
	conns[from + '_' + to] = yourConnection;
	return yourConnection;
}

// 删除p2p连接
function delConnection(from, to) {
	let testName = from + '_' + to;
	if (conns[testName]) {
		delete conns[testName];
	}
	testName = to + '_' + from;
	if (conns[testName]) {
		delete conns[testName];
	}
}
