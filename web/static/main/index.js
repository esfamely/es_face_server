class IndexTab extends React.Component {
    constructor(props) {
        super(props);
        this.state = {tab: 0};
    }

    handleClick(e, i) {
        e.preventDefault();
        this.setState({tab: i});
    }

    render() {
        const tabs = tabList.map((tab, i) => {
            const key = 'tab' + i;
            const tabShow1 = this.state.tab == i ? ('[' + tab + ']') : tab;
            const nbsp = <span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>;
            const tabShow2 = <a href="" onClick={(e) => this.handleClick(e, i)}>{tabShow1}</a>;
            return <span key={key}>{nbsp}{tabShow2}</span>
        });

        return (
            <div>{tabs}</div>
        );
    }
}

ReactDOM.render(
    <CollectUI />,
    document.getElementById('root')
);
