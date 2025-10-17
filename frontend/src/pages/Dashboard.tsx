import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Statistic, Table, Tag, Button, message } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, ReloadOutlined } from '@ant-design/icons';
import { portfolioAPI, dataAPI } from '../services/api';

interface Portfolio {
  id: number;
  name: string;
  current_value: number;
  initial_capital: number;
  optimizer_type: string;
  status: string;
  created_at: string;
}

const Dashboard: React.FC = () => {
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [loading, setLoading] = useState(false);
  const [dataStatus, setDataStatus] = useState<any>(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [portfolioRes, dataRes] = await Promise.all([
        portfolioAPI.list(),
        dataAPI.status(),
      ]);
      setPortfolios(portfolioRes.data);
      setDataStatus(dataRes.data);
    } catch (error) {
      message.error('Failed to load data');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  // Calculate aggregate metrics
  const totalValue = portfolios.reduce((sum, p) => sum + (p.current_value || 0), 0);
  const totalCapital = portfolios.reduce((sum, p) => sum + p.initial_capital, 0);
  const totalReturn = totalCapital > 0 ? ((totalValue - totalCapital) / totalCapital) * 100 : 0;

  const columns = [
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Value',
      dataIndex: 'current_value',
      key: 'value',
      render: (val: number) => `$${val?.toLocaleString(undefined, { minimumFractionDigits: 2 })}`,
    },
    {
      title: 'Return',
      key: 'return',
      render: (_: any, record: Portfolio) => {
        const ret = ((record.current_value - record.initial_capital) / record.initial_capital) * 100;
        const color = ret >= 0 ? 'green' : 'red';
        const icon = ret >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />;
        return (
          <span style={{ color }}>
            {icon} {ret.toFixed(2)}%
          </span>
        );
      },
    },
    {
      title: 'Optimizer',
      dataIndex: 'optimizer_type',
      key: 'optimizer',
      render: (opt: string) => <Tag color="blue">{opt.toUpperCase()}</Tag>,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'active' ? 'green' : 'default'}>
          {status.toUpperCase()}
        </Tag>
      ),
    },
  ];

  return (
    <div>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1>Dashboard</h1>
        <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>
          Refresh
        </Button>
      </div>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card>
            <Statistic
              title="Total Portfolio Value"
              value={totalValue}
              precision={2}
              prefix="$"
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="Total Return"
              value={totalReturn}
              precision={2}
              suffix="%"
              valueStyle={{ color: totalReturn >= 0 ? '#3f8600' : '#cf1322' }}
              prefix={totalReturn >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic title="Active Portfolios" value={portfolios.filter(p => p.status === 'active').length} />
          </Card>
        </Col>
      </Row>

      {dataStatus && (
        <Card title="Data Status" style={{ marginBottom: 24 }} size="small">
          <Row gutter={16}>
            <Col span={8}>
              <Statistic
                title="Latest Price Data"
                value={dataStatus.latest_price_date || 'N/A'}
                valueStyle={{ fontSize: 16 }}
              />
            </Col>
            <Col span={8}>
              <Statistic
                title="Days Old"
                value={dataStatus.days_old || 0}
                valueStyle={{ fontSize: 16, color: dataStatus.days_old > 2 ? '#cf1322' : '#3f8600' }}
              />
            </Col>
            <Col span={8}>
              <Statistic title="ETFs Tracked" value={dataStatus.num_etfs || 0} valueStyle={{ fontSize: 16 }} />
            </Col>
          </Row>
        </Card>
      )}

      <Card title="Portfolios">
        <Table
          dataSource={portfolios}
          columns={columns}
          rowKey="id"
          loading={loading}
          pagination={false}
        />
      </Card>
    </div>
  );
};

export default Dashboard;
