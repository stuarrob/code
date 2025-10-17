import React, { useEffect, useState } from 'react';
import { Card, Button, Modal, Form, Input, Select, InputNumber, Table, message, Space, Tag, Descriptions, Statistic, Row, Col } from 'antd';
import { PlusOutlined, ReloadOutlined } from '@ant-design/icons';
import { portfolioAPI, positionAPI, tradeAPI, factorAPI } from '../services/api';

interface Portfolio {
  id: number;
  name: string;
  current_value: number;
  initial_capital: number;
  optimizer_type: string;
  num_positions: number;
  status: string;
}

const PortfolioManager: React.FC = () => {
  const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [detailsVisible, setDetailsVisible] = useState(false);
  const [tradeModalVisible, setTradeModalVisible] = useState(false);
  const [selectedPortfolio, setSelectedPortfolio] = useState<Portfolio | null>(null);
  const [positions, setPositions] = useState<any[]>([]);
  const [driftInfo, setDriftInfo] = useState<any>(null);
  const [form] = Form.useForm();
  const [tradeForm] = Form.useForm();

  useEffect(() => {
    loadPortfolios();
  }, []);

  const loadPortfolios = async () => {
    setLoading(true);
    try {
      const response = await portfolioAPI.list();
      setPortfolios(response.data);
    } catch (error) {
      message.error('Failed to load portfolios');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async (values: any) => {
    const hide = message.loading('Creating portfolio and generating positions...', 0);
    try {
      // 1. Create the portfolio
      const portfolioResponse = await portfolioAPI.create({
        ...values,
        is_paper_trading: true,
      });
      const portfolio = portfolioResponse.data;

      // 2. Get factor recommendations
      const recsResponse = await factorAPI.recommendations(values.num_positions);
      const recommendations = recsResponse.data;

      if (recommendations && recommendations.length > 0) {
        // 3. Execute initial trades to establish positions
        // Backend will fetch real prices and calculate shares
        for (const rec of recommendations) {
          const dollarAmount = values.initial_capital * rec.target_weight; // Amount to invest in this position
          await tradeAPI.execute({
            portfolio_id: portfolio.id,
            ticker: rec.ticker,
            side: 'buy',
            dollar_amount: dollarAmount, // Backend will convert to shares using real price
            price: 100, // Placeholder that triggers backend to fetch real price
          });
        }
        message.success(`Portfolio created with ${recommendations.length} positions from AQR strategy!`);
      } else {
        message.warning('Portfolio created but no factor recommendations available. Please load factor scores.');
      }

      hide();
      setModalVisible(false);
      form.resetFields();
      loadPortfolios();
    } catch (error: any) {
      hide();
      message.error(error.response?.data?.detail || 'Failed to create portfolio');
      console.error('Portfolio creation error:', error);
    }
  };

  const handleView = async (portfolio: Portfolio) => {
    setSelectedPortfolio(portfolio);
    setDetailsVisible(true);
    try {
      const [posResponse, driftResponse] = await Promise.all([
        positionAPI.list(portfolio.id),
        positionAPI.getDrift(portfolio.id)
      ]);
      setPositions(posResponse.data);
      setDriftInfo(driftResponse.data);
    } catch (error) {
      console.error('Failed to load positions', error);
      setPositions([]);
      setDriftInfo(null);
    }
  };

  const handleExecuteTrade = async (values: any) => {
    if (!selectedPortfolio) return;

    try {
      await tradeAPI.execute({
        portfolio_id: selectedPortfolio.id,
        ...values,
      });
      message.success(`Trade executed: ${values.side.toUpperCase()} ${values.quantity} ${values.ticker}`);
      setTradeModalVisible(false);
      tradeForm.resetFields();

      // Reload positions
      const response = await positionAPI.list(selectedPortfolio.id);
      setPositions(response.data);
      loadPortfolios(); // Refresh portfolio values
    } catch (error: any) {
      message.error(error.response?.data?.detail || 'Failed to execute trade');
    }
  };

  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 60,
    },
    {
      title: 'Name',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Current Value',
      dataIndex: 'current_value',
      key: 'value',
      render: (val: number) => `$${val?.toLocaleString(undefined, { minimumFractionDigits: 2 })}`,
    },
    {
      title: 'Initial Capital',
      dataIndex: 'initial_capital',
      key: 'capital',
      render: (val: number) => `$${val?.toLocaleString(undefined, { minimumFractionDigits: 2 })}`,
    },
    {
      title: 'Optimizer',
      dataIndex: 'optimizer_type',
      key: 'optimizer',
      render: (opt: string) => <Tag color="blue">{opt.toUpperCase()}</Tag>,
    },
    {
      title: 'Positions',
      dataIndex: 'num_positions',
      key: 'positions',
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
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: Portfolio) => (
        <Space>
          <Button size="small" type="link" onClick={() => handleView(record)}>View</Button>
          <Button size="small" type="link" danger>Close</Button>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between' }}>
        <h1>Portfolio Manager</h1>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={loadPortfolios} loading={loading}>
            Refresh
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalVisible(true)}>
            Create Portfolio
          </Button>
        </Space>
      </div>

      <Card>
        <Table
          dataSource={portfolios}
          columns={columns}
          rowKey="id"
          loading={loading}
        />
      </Card>

      <Modal
        title="Create New Portfolio"
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false);
          form.resetFields();
        }}
        footer={null}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreate}
          initialValues={{
            optimizer_type: 'mvo',
            num_positions: 20,
            initial_capital: 1000000,
          }}
        >
          <Form.Item
            label="Portfolio Name"
            name="name"
            rules={[{ required: true, message: 'Please enter portfolio name' }]}
          >
            <Input placeholder="My Portfolio" />
          </Form.Item>

          <Form.Item
            label="Optimizer"
            name="optimizer_type"
            rules={[{ required: true }]}
          >
            <Select>
              <Select.Option value="mvo">Mean-Variance (MVO)</Select.Option>
              <Select.Option value="rank_based">Rank-Based</Select.Option>
              <Select.Option value="minvar">Minimum Variance</Select.Option>
              <Select.Option value="simple">Simple Equal-Weight</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item
            label="Number of Positions"
            name="num_positions"
            rules={[{ required: true }]}
          >
            <InputNumber min={5} max={50} style={{ width: '100%' }} />
          </Form.Item>

          <Form.Item
            label="Initial Capital"
            name="initial_capital"
            rules={[{ required: true }]}
          >
            <InputNumber
              min={10000}
              max={100000000}
              step={100000}
              formatter={(value) => `$ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
              parser={(value) => value!.replace(/\$\s?|(,*)/g, '') as any}
              style={{ width: '100%' }}
            />
          </Form.Item>

          <Form.Item>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => setModalVisible(false)}>Cancel</Button>
              <Button type="primary" htmlType="submit">
                Create
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="Portfolio Details"
        open={detailsVisible}
        onCancel={() => setDetailsVisible(false)}
        footer={[
          <Button key="close" type="primary" onClick={() => setDetailsVisible(false)}>
            Close
          </Button>
        ]}
        width={900}
      >
        {selectedPortfolio && (
          <>
            <Row gutter={16} style={{ marginBottom: 24 }}>
              <Col span={8}>
                <Card>
                  <Statistic
                    title="Current Value"
                    value={parseFloat(selectedPortfolio.current_value as any)}
                    precision={2}
                    prefix="$"
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card>
                  <Statistic
                    title="Initial Capital"
                    value={parseFloat(selectedPortfolio.initial_capital as any)}
                    precision={2}
                    prefix="$"
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card>
                  <Statistic
                    title="Return"
                    value={((parseFloat(selectedPortfolio.current_value as any) - parseFloat(selectedPortfolio.initial_capital as any)) / parseFloat(selectedPortfolio.initial_capital as any)) * 100}
                    precision={2}
                    suffix="%"
                    valueStyle={{
                      color: parseFloat(selectedPortfolio.current_value as any) >= parseFloat(selectedPortfolio.initial_capital as any) ? '#3f8600' : '#cf1322'
                    }}
                  />
                </Card>
              </Col>
            </Row>

            <Descriptions bordered column={2} style={{ marginBottom: 24 }}>
              <Descriptions.Item label="Portfolio ID">{selectedPortfolio.id}</Descriptions.Item>
              <Descriptions.Item label="Name">{selectedPortfolio.name}</Descriptions.Item>
              <Descriptions.Item label="Optimizer">{selectedPortfolio.optimizer_type.toUpperCase()}</Descriptions.Item>
              <Descriptions.Item label="Target Positions">{selectedPortfolio.num_positions}</Descriptions.Item>
              <Descriptions.Item label="Status">
                <Tag color={selectedPortfolio.status === 'active' ? 'green' : 'default'}>
                  {selectedPortfolio.status.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              {driftInfo && (
                <Descriptions.Item label="Rebalancing Status">
                  <Tag color={driftInfo.needs_rebalancing ? 'orange' : 'green'}>
                    {driftInfo.needs_rebalancing ? `NEEDS REBALANCING (${(driftInfo.max_drift * 100).toFixed(2)}% drift)` : 'OK'}
                  </Tag>
                </Descriptions.Item>
              )}
            </Descriptions>

            <Card title="Positions" size="small">
              {positions.length > 0 ? (
                <Table
                  dataSource={positions}
                  columns={[
                    { title: 'Ticker', dataIndex: 'ticker', key: 'ticker' },
                    { title: 'Shares', dataIndex: 'shares', key: 'shares', render: (val: any) => parseFloat(val)?.toFixed(2) },
                    { title: 'Entry Price', dataIndex: 'entry_price', key: 'entry', render: (val: any) => `$${parseFloat(val)?.toFixed(2)}` },
                    { title: 'Current Price', dataIndex: 'current_price', key: 'current', render: (val: any) => `$${parseFloat(val || 0)?.toFixed(2)}` },
                  ]}
                  pagination={false}
                  size="small"
                />
              ) : (
                <div style={{ textAlign: 'center', padding: 40 }}>
                  <p style={{ color: '#999', fontSize: 16, marginBottom: 10 }}>
                    No positions in this portfolio yet.
                  </p>
                  <p style={{ color: '#666', fontSize: 14 }}>
                    Positions should have been created automatically. If this portfolio was just created, please refresh to see updated positions.
                  </p>
                </div>
              )}
            </Card>
          </>
        )}
      </Modal>

      <Modal
        title="Execute Trade"
        open={tradeModalVisible}
        onCancel={() => {
          setTradeModalVisible(false);
          tradeForm.resetFields();
        }}
        footer={null}
      >
        <Form
          form={tradeForm}
          layout="vertical"
          onFinish={handleExecuteTrade}
          initialValues={{
            side: 'buy',
            quantity: 10,
            price: 100
          }}
        >
          <Form.Item
            label="Ticker Symbol"
            name="ticker"
            rules={[
              { required: true, message: 'Please enter ticker' },
              { pattern: /^[A-Z]{1,5}$/, message: 'Enter valid ticker (e.g., QQQ)' }
            ]}
          >
            <Input placeholder="QQQ" style={{ textTransform: 'uppercase' }} />
          </Form.Item>

          <Form.Item
            label="Side"
            name="side"
            rules={[{ required: true }]}
          >
            <Select>
              <Select.Option value="buy">Buy</Select.Option>
              <Select.Option value="sell">Sell</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item
            label="Quantity"
            name="quantity"
            rules={[{ required: true, message: 'Please enter quantity' }]}
          >
            <InputNumber min={1} max={10000} style={{ width: '100%' }} />
          </Form.Item>

          <Form.Item
            label="Price per Share"
            name="price"
            rules={[{ required: true, message: 'Please enter price' }]}
          >
            <InputNumber
              min={0.01}
              max={100000}
              step={0.01}
              formatter={(value) => `$ ${value}`}
              parser={(value) => value!.replace(/\$\s?/g, '') as any}
              style={{ width: '100%' }}
            />
          </Form.Item>

          <Form.Item>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => setTradeModalVisible(false)}>Cancel</Button>
              <Button type="primary" htmlType="submit">
                Execute Trade
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default PortfolioManager;
