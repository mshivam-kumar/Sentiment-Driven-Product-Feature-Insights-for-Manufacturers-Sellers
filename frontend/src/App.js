import React, { useState } from 'react';
import './App.css';
import Header from './components/Header';
import ProductAnalysis from './components/ProductAnalysis';
import FeatureSearch from './components/FeatureSearch';
import ChatAssistant from './components/ChatAssistant';
import TestProductAnalysis from './components/TestProductAnalysis';
import Footer from './components/Footer';

function App() {
  const [activeTab, setActiveTab] = useState('product');

  const renderActiveComponent = () => {
    switch (activeTab) {
      case 'product':
        return <ProductAnalysis />;
      case 'feature':
        return <FeatureSearch />;
      case 'chat':
        return <ChatAssistant />;
      case 'test':
        return <TestProductAnalysis />;
      default:
        return <ProductAnalysis />;
    }
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="App">
      <Header />
      
      <div className="main-container">
        <div className="tab-navigation">
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            <button 
              className={`tab-button ${activeTab === 'product' ? 'active' : ''}`}
              onClick={() => setActiveTab('product')}
            >
              📱 Product Analysis
            </button>
            <button 
              className={`tab-button ${activeTab === 'feature' ? 'active' : ''}`}
              onClick={() => setActiveTab('feature')}
            >
              🔍 Feature Search
            </button>
            <button 
              className={`tab-button ${activeTab === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveTab('chat')}
            >
              🤖 AI Chat Assistant
            </button>
          </div>
          <button 
            className="btn btn-secondary"
            onClick={handlePrint}
            style={{ 
              marginLeft: 'auto',
              fontSize: '14px',
              padding: '8px 16px',
              display: 'flex',
              alignItems: 'center',
              gap: '5px'
            }}
          >
            🖨️ Print Page
          </button>
        </div>

        <div className="content">
          {renderActiveComponent()}
        </div>
      </div>

      <Footer />
    </div>
  );
}

export default App;