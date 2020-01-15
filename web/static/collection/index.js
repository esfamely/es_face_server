window.onload = () => {
	let status = document.querySelector('#status');
	let leftDiv = document.querySelector('#leftDiv');
	let collectText = document.querySelector('#collectText');
	let collectButton = document.querySelector('#collectButton');
	let predictButton = document.querySelector('#predictButton');
	let collectXButton = document.querySelector('#collectXButton');
	let collectPButton = document.querySelector('#collectPButton');
	let frameDiv = document.querySelector('#frameDiv');
	let cameraIframe = document.querySelector('#cameraIframe');

	leftDiv.style.width = (document.body.clientWidth - cameraIframe.width) / 2 + 'px';

	status.style.display = 'none';
	leftDiv.style.display = 'inline';
	frameDiv.style.display = 'inline';

	collectButton.addEventListener('click', event => {
		if (!collectText.value || collectText.value == '') {
			return;
		}
		cameraFrame.window.toCollect(collectText.value);
	});
	predictButton.addEventListener('click', event => {
		cameraFrame.window.toPredict();
	});
	collectXButton.addEventListener('click', event => {
		collectXButton.style.display = 'none';
		collectPButton.style.display = 'inline';
		cameraFrame.window.toCollectX(3);
	});
	collectPButton.addEventListener('click', event => {
		collectXButton.style.display = 'inline';
		collectPButton.style.display = 'none';
		cameraFrame.window.toCollectX(-1);
	});
};

function setFPS(fps) {
	let fpsDiv = document.querySelector('#fpsDiv');
	fpsDiv.innerHTML = '当前FPS：' + fps;
}

function setCollectResult(result) {
	if (result.uid) {
		alert('你是 ' + result.uid);
	}
	let collectDiv = document.querySelector('#collectDiv');
	collectDiv.innerHTML = '采集结果：采集成功，得到采集id：' + result.cid;
}
