import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import { API_ENDPOINTS } from '../config/api';

const ProductAnalysis = () => {
  const [asin, setAsin] = useState(() => {
    const saved = localStorage.getItem('productAnalysis_asin');
    return saved || '';
  });
  const [window, setWindow] = useState(() => {
    const saved = localStorage.getItem('productAnalysis_window');
    return saved || '';
  });
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(() => {
    const saved = localStorage.getItem('productAnalysis_results');
    return saved ? JSON.parse(saved) : null;
  });
  const [error, setError] = useState(null);

  // Example ASINs for quick testing
  const exampleAsins = [
    'B08JTNQFZY',
    'B00YQ6X8EO',
    'B07XJ8C8F5',
    'B08N5WRWNW',
    'B07ZPKN6YR'
  ];

  const timeWindows = [
    { value: '', label: 'All Time' },
    { value: '10y', label: '10 Years' },
    { value: '1y', label: '1 Year' },
    { value: '6m', label: '6 Months' },
    { value: '3m', label: '3 Months' },
    { value: '1m', label: '1 Month' }
  ];

  // Save data to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('productAnalysis_asin', asin);
  }, [asin]);

  useEffect(() => {
    localStorage.setItem('productAnalysis_window', window);
  }, [window]);

  useEffect(() => {
    if (results) {
      localStorage.setItem('productAnalysis_results', JSON.stringify(results));
    }
  }, [results]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!asin.trim()) {
      setError('Please enter a valid ASIN');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const params = new URLSearchParams();
      if (window) params.append('window', window);

      console.log(`Making API call to: ${API_ENDPOINTS.PRODUCT_ANALYSIS}/${asin}?${params}`);
      const response = await fetch(`${API_ENDPOINTS.PRODUCT_ANALYSIS}/${asin}?${params}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('API Response:', data);

      if (data.success) {
        // Backend now sends features as an array, so use it directly
        const features = data.data.features || [];
        console.log('🔍 Raw features array:', features);
        console.log('🔍 Features count:', features.length);
        
        // Features are already in the correct format from backend
        const transformedData = {
          asin: data.data.asin,
          overall_sentiment: data.data.overall_sentiment,
          total_reviews: data.data.total_reviews,
          features: features
        };
        
        console.log('🔄 Transformed data:', transformedData);
        console.log('🔄 Features count in transformed data:', transformedData.features.length);
        setResults(transformedData);
      } else {
        setError(data.detail || 'Failed to fetch product analysis');
      }
    } catch (err) {
      setError('Network error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (exampleAsin) => {
    setAsin(exampleAsin);
  };

  return (
    <div>
      <h2>📱 Product Analysis</h2>
      <p>Enter an Amazon ASIN to get comprehensive sentiment analysis</p>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="asin">Amazon ASIN:</label>
            <input
              type="text"
              id="asin"
              value={asin}
              onChange={(e) => setAsin(e.target.value)}
              placeholder="e.g., B08JTNQFZY"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="window">Time Window:</label>
            <select
              id="window"
              value={window}
              onChange={(e) => setWindow(e.target.value)}
            >
              {timeWindows.map(tw => (
                <option key={tw.value} value={tw.value}>{tw.label}</option>
              ))}
            </select>
          </div>

          <button type="submit" className="btn" disabled={loading}>
            {loading ? 'Analyzing...' : 'Analyze Product'}
          </button>
        </form>

        <div style={{ marginTop: '20px' }}>
          <h4>💡 Example ASINs:</h4>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            {exampleAsins.map(exampleAsin => (
              <button
                key={exampleAsin}
                className="btn btn-secondary"
                onClick={() => handleExampleClick(exampleAsin)}
                style={{ fontSize: '14px', padding: '8px 16px' }}
              >
                {exampleAsin}
              </button>
            ))}
          </div>
        </div>
      </div>

      {error && (
        <div className="card" style={{ background: '#fee', border: '1px solid #fcc' }}>
          <h3 style={{ color: '#c33', margin: '0 0 10px 0' }}>❌ Error</h3>
          <p style={{ margin: 0, color: '#c33' }}>{error}</p>
        </div>
      )}

      {loading && (
        <div className="loading">
          <div className="spinner"></div>
          <p>Analyzing product sentiment...</p>
        </div>
      )}

      {results && (
        <div>
          <div className="card">
            <h3>📊 Analysis Results</h3>
            <div className="metrics">
              <div className="metric">
                <h4>Overall Sentiment</h4>
                <p className="value" style={{ 
                  color: results.overall_sentiment > 0.5 ? '#10b981' : results.overall_sentiment < -0.5 ? '#ef4444' : '#6b7280',
                  fontSize: '1.5rem',
                  fontWeight: 'bold'
                }}>
                  {results.overall_sentiment?.toFixed(2) || 'N/A'}
                </p>
                <div style={{ 
                  width: '100%', 
                  height: '8px', 
                  background: '#e5e7eb', 
                  borderRadius: '4px',
                  marginTop: '5px'
                }}>
                  <div style={{
                    width: `${Math.max(0, Math.min(100, (results.overall_sentiment + 1) * 50))}%`,
                    height: '100%',
                    background: results.overall_sentiment > 0.5 ? '#10b981' : results.overall_sentiment < -0.5 ? '#ef4444' : '#6b7280',
                    borderRadius: '4px',
                    transition: 'width 0.3s ease'
                  }}></div>
                </div>
              </div>
              <div className="metric">
                <h4>Total Reviews</h4>
                <p className="value" style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#3b82f6' }}>
                  {results.total_reviews || 0}
                </p>
              </div>
              <div className="metric">
                <h4>Features Analyzed</h4>
                <p className="value" style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#8b5cf6' }}>
                  {results.features?.length || 0}
                </p>
              </div>
              <div className="metric">
                <h4>Product ASIN</h4>
                <p className="value" style={{ 
                  fontSize: '1.2rem', 
                  fontFamily: 'monospace',
                  background: '#f3f4f6',
                  padding: '8px',
                  borderRadius: '4px',
                  border: '1px solid #d1d5db'
                }}>
                  {results.asin || asin}
                </p>
              </div>
            </div>
          </div>

          {results.features && results.features.length > 0 && (
            <>
              {/* Charts Section */}
              <div className="card">
                <h3>📊 Product Sentiment Charts</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '20px', marginBottom: '20px' }}>
                  {/* Feature Sentiment Bar Chart */}
                  <div>
                    <h4 style={{ textAlign: 'center', marginBottom: '15px' }}>Feature Sentiment Scores</h4>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={results.features.map(feature => ({
                        name: feature.feature.replace(/_/g, ' ').substring(0, 12),
                        sentiment: feature.sentiment,
                        count: feature.count
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                        <YAxis />
                        <Tooltip 
                          formatter={(value, name) => [value, name === 'sentiment' ? 'Sentiment Score' : 'Mentions']}
                          labelFormatter={(label) => `Feature: ${label}`}
                        />
                        <Bar dataKey="sentiment" fill="#3b82f6" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Feature Mentions Pie Chart */}
                  <div>
                    <h4 style={{ textAlign: 'center', marginBottom: '15px' }}>Feature Mentions Distribution</h4>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={results.features.map(feature => ({
                            name: feature.feature.replace(/_/g, ' '),
                            value: feature.count,
                            color: feature.sentiment > 0.1 ? '#10b981' : feature.sentiment < -0.1 ? '#ef4444' : '#6b7280'
                          }))}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, value, percent }) => `${name}: ${value} (${(percent * 100).toFixed(0)}%)`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {results.features.map((feature, index) => (
                            <Cell key={`cell-${index}`} fill={feature.sentiment > 0.1 ? '#10b981' : feature.sentiment < -0.1 ? '#ef4444' : '#6b7280'} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Feature Cards */}
              <div className="card">
                <h3>🔍 Feature Analysis</h3>
              <div style={{ display: 'grid', gap: '20px', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))' }}>
                {results.features.map((feature, index) => (
                  <div key={index} style={{ 
                    padding: '20px', 
                    border: '1px solid #e1e5e9', 
                    borderRadius: '12px',
                    background: feature.sentiment > 0 ? '#f0fdf4' : feature.sentiment < 0 ? '#fef2f2' : '#f8fafc',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
                    transition: 'transform 0.2s ease, box-shadow 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.05)';
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                      <h4 style={{ 
                        margin: 0, 
                        color: '#1f2937',
                        fontSize: '1.1rem',
                        fontWeight: '600',
                        textTransform: 'capitalize'
                      }}>
                        {feature.feature.replace(/_/g, ' ')}
                      </h4>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <div style={{
                          width: '60px',
                          height: '8px',
                          background: '#e5e7eb',
                          borderRadius: '4px',
                          overflow: 'hidden'
                        }}>
                          <div style={{
                            width: `${Math.max(0, Math.min(100, (feature.sentiment + 1) * 50))}%`,
                            height: '100%',
                            background: feature.sentiment > 0.1 ? '#10b981' : feature.sentiment < -0.1 ? '#ef4444' : '#6b7280',
                            transition: 'width 0.3s ease'
                          }}></div>
                        </div>
                        <span style={{ 
                          padding: '6px 12px', 
                          borderRadius: '20px', 
                          background: feature.sentiment > 0.1 ? '#10b981' : feature.sentiment < -0.1 ? '#ef4444' : '#6b7280',
                          color: 'white',
                          fontSize: '12px',
                          fontWeight: '600'
                        }}>
                          {feature.sentiment > 0.1 ? 'Positive' : feature.sentiment < -0.1 ? 'Negative' : 'Neutral'}
                        </span>
                      </div>
                    </div>
                    
                    <div style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      alignItems: 'center',
                      marginBottom: '15px',
                      padding: '10px',
                      background: 'white',
                      borderRadius: '8px',
                      border: '1px solid #e5e7eb'
                    }}>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#3b82f6' }}>
                          {feature.sentiment?.toFixed(2)}
                        </div>
                        <div style={{ fontSize: '12px', color: '#666' }}>Sentiment Score</div>
                      </div>
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#8b5cf6' }}>
                          {feature.count}
                        </div>
                        <div style={{ fontSize: '12px', color: '#666' }}>Reviews</div>
                      </div>
                    </div>

                    {feature.snippets && feature.snippets.length > 0 && (
                      <div>
                        <div style={{ 
                          fontSize: '14px', 
                          fontWeight: '600', 
                          color: '#374151',
                          marginBottom: '10px'
                        }}>
                          💬 Sample Reviews:
                        </div>
                        {feature.snippets.slice(0, 2).map((snippet, idx) => (
                          <div key={idx} style={{ 
                            fontSize: '13px', 
                            color: '#4b5563', 
                            marginBottom: '8px', 
                            padding: '12px', 
                            background: 'white', 
                            borderRadius: '8px',
                            border: '1px solid #e5e7eb',
                            borderLeft: `4px solid ${feature.sentiment > 0.1 ? '#10b981' : feature.sentiment < -0.1 ? '#ef4444' : '#6b7280'}`,
                            lineHeight: '1.5'
                          }}>
                            "{snippet.substring(0, 150)}{snippet.length > 150 ? '...' : ''}"
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default ProductAnalysis;
