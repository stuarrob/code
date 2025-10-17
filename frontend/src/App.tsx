import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Layout, Menu } from 'antd';
import { DashboardOutlined, FundOutlined, LineChartOutlined } from '@ant-design/icons';
import Dashboard from './pages/Dashboard';
import PortfolioManager from './pages/PortfolioManager';
import './App.css';

const { Header, Content, Sider } = Layout;

const App: React.FC = () => {
  return (
    <Router>
      <Layout style={{ minHeight: '100vh' }}>
        <Header style={{
          position: 'fixed',
          zIndex: 1000,
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          background: '#001529'
        }}>
          <div style={{ color: 'white', fontSize: '20px', fontWeight: 'bold' }}>
            📈 ETF Trader
          </div>
        </Header>
        <Layout style={{ marginTop: '64px' }}>
          <Sider width={200} style={{ background: '#fff' }}>
            <Menu
              mode="inline"
              defaultSelectedKeys={['1']}
              style={{ height: '100%', borderRight: 0 }}
            >
              <Menu.Item key="1" icon={<DashboardOutlined />}>
                <Link to="/">Dashboard</Link>
              </Menu.Item>
              <Menu.Item key="2" icon={<FundOutlined />}>
                <Link to="/portfolios">Portfolios</Link>
              </Menu.Item>
              <Menu.Item key="3" icon={<LineChartOutlined />}>
                <Link to="/factors">Factors</Link>
              </Menu.Item>
            </Menu>
          </Sider>
          <Layout style={{ padding: '24px' }}>
            <Content
              style={{
                padding: 24,
                margin: 0,
                minHeight: 280,
                background: '#fff',
              }}
            >
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/portfolios" element={<PortfolioManager />} />
                <Route path="/factors" element={<div>Factor Analysis (Coming Soon)</div>} />
              </Routes>
            </Content>
          </Layout>
        </Layout>
      </Layout>
    </Router>
  );
};

export default App;
