//---------------
//计算机视觉工具
//---------------

// 计算两图像的L2相对距离
function distance_l2_relative(img1, img2) {
	if (img1.rows != img2.rows || img1.cols != img2.cols) {
		return -1;
	}
	// 计算：欧氏距离平方 / 3通道像素总数
	let sum = 0;
	for (let i=0; i<img1.rows; i++) {
		for (let j=0; j<img1.cols; j++) {
			let c1 = img1.ucharPtr(i, j);
			let c2 = img2.ucharPtr(i, j);
			for (let k=0; k<3; k++) {
				let d = c1[k] - c2[k];
				sum += d * d;
			}
		}
	}
	return sum / (img1.rows * img1.cols * 3);
}

// 计算矩形面积
function area_rect(rect) {
	return rect.width * rect.height;
}

// 找到最大的矩形
function max_rect(rects) {
	if (rects.size() < 1) {
		return null;
	}
	let max_r = rects.get(0);
	let max_a = area_rect(max_r);
	for (let i=1; i<rects.size(); i++) {
		let rect = rects.get(i);
		let area = area_rect(rect);
		if (area > max_a) {
			max_r = rect;
			max_a = area;
		}
	}
	return max_r;
}

// 两矩形以p比例相加
function add_rect(rect1, rect2, p) {
	let x = Math.ceil(p*rect1.x + (1 - p)*rect2.x);
	let y = Math.ceil(p*rect1.y + (1 - p)*rect2.y);
	let width = Math.ceil(p*rect1.width + (1 - p)*rect2.width);
	let height = Math.ceil(p*rect1.height + (1 - p)*rect2.height);
	return {x: x, y: y, width: width, height: height};
}
