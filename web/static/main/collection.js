class StartupButton extends React.Component {
    constructor(props) {
        super(props);

        this.handleClick = this.handleClick.bind(this);
    }

    handleClick() {
        fetch(clientIp + '/face/startup', {
            method: 'GET',
            mode: 'cors'
        })
        .then(res => res.json())
        .then(
            (result) => {
                //alert(result.r);
            },
            (error) => {
                alert(error);
            }
        )
    }

    render() {
        return (
            <div>
                <button onClick={this.handleClick}>开启摄像头</button>
            </div>
        );
    }
}

class TrainButton extends React.Component {
    constructor(props) {
        super(props);
    }

    render() {
        return (
            <button>训练</button>
        );
    }
}

class CollectInput extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            showInput: false,
            uid: '',
            cid: ''
        };

        this.handleChange = this.handleChange.bind(this);
        this.handleClick = this.handleClick.bind(this);
    }

    handleChange(event) {
        this.setState({uid: event.target.value});
    }

    handleClick0(cid) {
        this.setState({showInput: true});
    }

    handleClick() {
        fetch(clientIp + '/face/collection/collect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                uid: this.state.uid
            })
        })
        .then(res => res.json())
        .then(
            (result) => {
                //alert('采集完成！');
                this.setState({cid: result.cid});
            },
            (error) => {
                alert(error);
            }
        )
    }

    handleClick2(cid) {
        this.setState({showInput: false, cid: cid});
    }

    render() {
        const { showInput, uid, cid } = this.state;

        let cr = '';
        if (cid != '') {
            cr = <CollectResult cid={cid} onClick={() => this.handleClick2('')} />;
        }

        if (showInput) {
            return (
                <span>
                    <label>
                      请选择采集人：
                      <input type="text" value={uid} onChange={this.handleChange} />
                    </label>
                    <button onClick={this.handleClick}>开始采集</button>
                    {cr}
                </span>
            );
        } else {
            return (
                <button onClick={() => this.handleClick0()}>采集</button>
            );
        }
    }
}

class CollectResult extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            isLoaded: false,
            error: null,
            items: []
        };
    }

    componentDidMount() {
        fetch(serverIp + '/face/collection/load_face_list3', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                cid: this.props.cid
            })
        })
        .then(res => res.json())
        .then(
            (result) => {
                this.setState({
                    isLoaded: true,
                    items: result
                });
            },
            (error) => {
                this.setState({
                    isLoaded: true,
                    error: error
                });
            }
        )
    }

    render() {
        const { isLoaded, error, items } = this.state;
        if (error) {
            return <div>加载异常: {error.message}</div>;
        } else if (!isLoaded) {
            return <div>加载中...</div>;
        } else {
            // 每行显示几个
            const ipr = 3;
            // 一维数组扩展为二维数组
            let itemss = new Array(Math.floor(items.length / ipr) + 1);
            for (let i=0; i<itemss.length; i++) {
                const start = i*ipr;
                const step = i == (itemss.length - 1) ? (items.length % ipr) : ipr;
                itemss[i] = Array.from(items.slice(start, start + step));
            }
            const nbsp = <span>&nbsp;</span>;
            return (
                <div>
                    <div>采集结果：</div>
                    <CollectResultOkButton onClick={() => this.props.onClick()} />
                    {itemss.map((items, i) => {
                        const key = 'img_row_' + i;
                        return <div key={key}>
                            {items.map(item => {
                                return <span key={item.id}>
                                    <span>
                                        <img src={item.img_url} />
                                    </span>
                                    {nbsp}
                                </span>
                            })}
                        </div>
                    })}
                </div>
            );
        }
    }
}

class CollectResultOkButton extends React.Component {
    constructor(props) {
        super(props);
    }

    render() {
        return (
            <div>
                <button onClick={() => this.props.onClick()}>确定</button>
            </div>
        );
    }
}

class FaceStat extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            isLoaded: false,
            error: null,
            result: {}
        };
    }

    componentDidMount() {
        fetch(serverIp + '/face/collection/load_face_stat')
        .then(res => res.json())
        .then(
            (result) => {
                this.setState({
                    isLoaded: true,
                    result: result
                });
            },
            (error) => {
                this.setState({
                    isLoaded: true,
                    error: error
                });
            }
        )
    }

    render() {
        const { isLoaded, error, result } = this.state;
        if (error) {
            return <div>加载异常: {error.message}</div>;
        } else if (!isLoaded) {
            return <div>加载中...</div>;
        } else {
            return (
                <div>
                    <p>采集人数：{result.cc1}</p>
                    <p>采集样本数：{result.cc2}</p>
                </div>
            );
        }
    }
}

class FaceList1 extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            isLoaded: false,
            error: null,
            items: []
        };
    }

    componentDidMount() {
        fetch(serverIp + '/face/collection/load_face_list1')
        .then(res => res.json())
        .then(
            (result) => {
                this.setState({
                    isLoaded: true,
                    items: result
                });
            },
            (error) => {
                this.setState({
                    isLoaded: true,
                    error: error
                });
            }
        )
    }

    render() {
        const { isLoaded, error, items } = this.state;
        if (error) {
            return <div>加载异常: {error.message}</div>;
        } else if (!isLoaded) {
            return <div>加载中...</div>;
        } else {
            // 每行显示几个
            const ipr = 2;
            // 一维数组扩展为二维数组
            let itemss = new Array(Math.floor(items.length / ipr) + 1);
            for (let i=0; i<itemss.length; i++) {
                const start = i*ipr;
                const step = i == (itemss.length - 1) ? (items.length % ipr) : ipr;
                itemss[i] = Array.from(items.slice(start, start + step));
            }
            return (
                <div>
                    {itemss.map((items, i) => {
                        const key = 'img_row_' + i;
                        return <div key={key}>
                            {items.map(item => {
                                return <span key={item.uid}
                                    onClick={() => this.props.onClick(item.uid)}>
                                    <span>
                                        <img src={item.img_url} />
                                    </span>
                                    <span>
                                        {item.un} 最近一次采集时间：{item.dt}
                                    </span>
                                </span>
                            })}
                        </div>
                    })}
                </div>
            );
        }
    }
}

class FaceList2 extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            isLoaded: false,
            error: null,
            items: []
        };
    }

    componentDidMount() {
        fetch(serverIp + '/face/collection/load_face_list2', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                uid: this.props.uid
            })
        })
        .then(res => res.json())
        .then(
            (result) => {
                this.setState({
                    isLoaded: true,
                    items: result
                });
            },
            (error) => {
                this.setState({
                    isLoaded: true,
                    error: error
                });
            }
        )
    }

    render() {
        const { isLoaded, error, items } = this.state;
        if (error) {
            return <div>加载异常: {error.message}</div>;
        } else if (!isLoaded) {
            return <div>加载中...</div>;
        } else {
            // 每行显示几个
            const ipr = 3;
            // 一维数组扩展为二维数组
            let itemss = new Array(Math.floor(items.length / ipr) + 1);
            for (let i=0; i<itemss.length; i++) {
                const start = i*ipr;
                const step = i == (itemss.length - 1) ? (items.length % ipr) : ipr;
                itemss[i] = Array.from(items.slice(start, start + step));
            }
            return (
                <div>
                    {itemss.map((items, i) => {
                        const key = 'img_row_' + i;
                        return <div key={key}>
                            {items.map(item => {
                                return <span key={item.id}>
                                    <span>
                                        <img src={item.img_url} />
                                    </span>
                                    <span>
                                        采集时间：{item.dt}
                                    </span>
                                </span>
                            })}
                        </div>
                    })}
                </div>
            );
        }
    }
}

class ToFaceList1Button extends React.Component {
    constructor(props) {
        super(props);
    }

    render() {
        return (
            <div>
                <button onClick={() => this.props.onClick()}>返回</button>
            </div>
        );
    }
}

class FaceInfo extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            uid: ''
        };
    }

    handleClick(uid) {
        this.setState({uid: uid});
    }

    render() {
        const { uid } = this.state;
        if (uid == '') {
            return (
                <div>
                    <FaceStat />
                    <FaceList1 onClick={uid => this.handleClick(uid)} />
                </div>
            );
        } else {
            return (
                <div>
                    <ToFaceList1Button onClick={() => this.handleClick('')} />
                    <FaceList2 uid={uid} />
                </div>
            );
        }
    }
}

class CollectUI extends React.Component {
    constructor(props) {
        super(props);
    }

    render() {
        const nbsp = <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>;
        return (
            <div>
                <StartupButton /><br />
                <div>
                    <CollectInput />{nbsp}<TrainButton />
                </div>
                <FaceInfo />
            </div>
        );
    }
}
