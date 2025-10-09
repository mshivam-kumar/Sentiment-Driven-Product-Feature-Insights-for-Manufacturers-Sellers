import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { API_ENDPOINTS } from '../config/api';

const FeatureSearch = () => {
  const [query, setQuery] = useState(() => {
    const saved = localStorage.getItem('featureSearch_query');
    return saved || '';
  });
  const [category, setCategory] = useState(() => {
    const saved = localStorage.getItem('featureSearch_category');
    return saved || '';
  });
  const [window, setWindow] = useState(() => {
    const saved = localStorage.getItem('featureSearch_window');
    return saved || '';
  });
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(() => {
    const saved = localStorage.getItem('featureSearch_results');
    return saved ? JSON.parse(saved) : null;
  });
  const [error, setError] = useState(null);

  // Example features for quick testing
  const exampleFeatures = [
    'battery',
    'quality',
    'design',
    'performance',
    'price',
    'durability',
    'comfort',
    'ease of use'
  ];

  const categories = [
    { value: '', label: 'All Categories' },
    { value: 'All_Beauty', label: 'Beauty' },
    { value: 'Electronics', label: 'Electronics' },
    { value: 'Home', label: 'Home & Kitchen' }
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
    localStorage.setItem('featureSearch_query', query);
  }, [query]);

  useEffect(() => {
    localStorage.setItem('featureSearch_category', category);
  }, [category]);

  useEffect(() => {
    localStorage.setItem('featureSearch_window', window);
  }, [window]);

  useEffect(() => {
    if (results) {
      localStorage.setItem('featureSearch_results', JSON.stringify(results));
    }
  }, [results]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) {
      setError('Please enter a search query');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const params = new URLSearchParams();
      params.append('query', query);
      if (category) params.append('category', category);
      if (window) params.append('window', window);
      params.append('limit', '20');

      const response = await fetch(`${API_ENDPOINTS.FEATURE_SEARCH}/search?${params}`);
      const data = await response.json();

      if (data.success) {
        // Transform the data to match expected format
        const results = data.data.results || [];
        const transformedResults = results.map(result => ({
          ...result,
          sentiment: result.score || 0,
          rating: result.rating || null,
          relevance_score: result.relevance_score || null,
          snippet: result.snippet || null
        }));
        
        const transformedData = {
          query: query,
          results: transformedResults,
          total: results.length
        };
        
        console.log('Feature search results:', transformedData);
        setResults(transformedData);
      } else {
        setError(data.detail || 'Failed to search features');
      }
    } catch (err) {
      setError('Network error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (exampleFeature) => {
    setQuery(exampleFeature);
  };

  return (
    <div>
      <h2>🔍 Feature Search</h2>
      <p>Search for specific features across all products and compare sentiment</p>

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="query">Feature Search Query:</label>
            <input
              type="text"
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., battery life, build quality, design"
              required
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
            <div className="form-group">
              <label htmlFor="category">Category:</label>
              <select
                id="category"
                value={category}
                onChange={(e) => setCategory(e.target.value)}
              >
                {categories.map(cat => (
                  <option key={cat.value} value={cat.value}>{cat.label}</option>
                ))}
              </select>
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
          </div>

          <button type="submit" className="btn" disabled={loading}>
            {loading ? 'Searching...' : 'Search Features'}
          </button>
        </form>

        <div style={{ marginTop: '20px' }}>
          <h4>💡 Example Features:</h4>
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            {exampleFeatures.map(exampleFeature => (
              <button
                key={exampleFeature}
                className="btn btn-secondary"
                onClick={() => handleExampleClick(exampleFeature)}
                style={{ fontSize: '14px', padding: '8px 16px' }}
              >
                {exampleFeature}
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
          <p>Searching for features...</p>
        </div>
      )}

      {results && (
        <div>
          <div className="card">
            <h3>📊 Search Results</h3>
            <div className="metrics">
              <div className="metric">
                <h4>Query</h4>
                <p className="value" style={{ fontSize: '1.2rem' }}>"{query}"</p>
              </div>
              <div className="metric">
                <h4>Results Found</h4>
                <p className="value">{results.results?.length || 0}</p>
              </div>
              <div className="metric">
                <h4>Category</h4>
                <p className="value" style={{ fontSize: '1rem' }}>{category || 'All'}</p>
              </div>
              <div className="metric">
                <h4>Time Window</h4>
                <p className="value" style={{ fontSize: '1rem' }}>{window || 'All Time'}</p>
              </div>
            </div>
          </div>

          {results.results && results.results.length > 0 && (
            <>
              {/* Charts Section */}
              <div className="card">
                <h3>📊 Sentiment Analysis Charts</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '20px', marginBottom: '20px' }}>
                  {/* Sentiment Distribution Chart */}
                  <div>
                    <h4 style={{ textAlign: 'center', marginBottom: '15px' }}>Sentiment Distribution</h4>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={[
                            { name: 'Positive', value: results.results.filter(r => r.sentiment > 0.1).length, color: '#10b981' },
                            { name: 'Negative', value: results.results.filter(r => r.sentiment < -0.1).length, color: '#ef4444' },
                            { name: 'Neutral', value: results.results.filter(r => r.sentiment >= -0.1 && r.sentiment <= 0.1).length, color: '#6b7280' }
                          ]}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, value, percent }) => `${name}: ${value} (${(percent * 100).toFixed(0)}%)`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {[
                            { name: 'Positive', value: results.results.filter(r => r.sentiment > 0.1).length, color: '#10b981' },
                            { name: 'Negative', value: results.results.filter(r => r.sentiment < -0.1).length, color: '#ef4444' },
                            { name: 'Neutral', value: results.results.filter(r => r.sentiment >= -0.1 && r.sentiment <= 0.1).length, color: '#6b7280' }
                          ].map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Sentiment Scores Bar Chart */}
                  <div>
                    <h4 style={{ textAlign: 'center', marginBottom: '15px' }}>Sentiment Scores by Product</h4>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={results.results.map((result, index) => ({
                        name: result.asin.substring(0, 8) + '...',
                        sentiment: result.sentiment,
                        count: result.count
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip 
                          formatter={(value, name) => [value, name === 'sentiment' ? 'Sentiment Score' : 'Mentions']}
                          labelFormatter={(label) => `ASIN: ${label}`}
                        />
                        <Bar dataKey="sentiment" fill="#3b82f6" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Results Cards */}
              <div className="card">
                <h3>🔍 Feature Analysis Results</h3>
                <div style={{ display: 'grid', gap: '20px', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))' }}>
                  {results.results.map((result, index) => (
                    <div key={index} style={{ 
                      padding: '20px', 
                      border: '1px solid #e1e5e9', 
                      borderRadius: '12px',
                      background: result.sentiment > 0.1 ? '#f0fdf4' : result.sentiment < -0.1 ? '#fef2f2' : '#f8fafc',
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
                        <div>
                          <h4 style={{ 
                            margin: '0 0 5px 0', 
                            color: '#1f2937',
                            fontSize: '1.1rem',
                            fontWeight: '600',
                            textTransform: 'capitalize'
                          }}>
                            {result.feature.replace(/_/g, ' ')}
                          </h4>
                          <p style={{ margin: 0, color: '#666', fontSize: '14px', fontFamily: 'monospace' }}>
                            ASIN: {result.asin}
                          </p>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                          <div style={{
                            width: '60px',
                            height: '8px',
                            background: '#e5e7eb',
                            borderRadius: '4px',
                            overflow: 'hidden'
                          }}>
                            <div style={{
                              width: `${Math.max(0, Math.min(100, (result.sentiment + 1) * 50))}%`,
                              height: '100%',
                              background: result.sentiment > 0.1 ? '#10b981' : result.sentiment < -0.1 ? '#ef4444' : '#6b7280',
                              transition: 'width 0.3s ease'
                            }}></div>
                          </div>
                          <span style={{ 
                            padding: '6px 12px', 
                            borderRadius: '20px', 
                            background: result.sentiment > 0.1 ? '#10b981' : result.sentiment < -0.1 ? '#ef4444' : '#6b7280',
                            color: 'white',
                            fontSize: '12px',
                            fontWeight: '600'
                          }}>
                            {result.sentiment > 0.1 ? 'Positive' : result.sentiment < -0.1 ? 'Negative' : 'Neutral'}
                          </span>
                        </div>
                      </div>
                      
                      <div style={{ 
                        display: 'flex', 
                        justifyContent: 'space-between', 
                        alignItems: 'center',
                        marginBottom: '15px',
                        padding: '15px',
                        background: 'white',
                        borderRadius: '8px',
                        border: '1px solid #e5e7eb'
                      }}>
                        <div style={{ textAlign: 'center' }}>
                          <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#3b82f6' }}>
                            {result.sentiment?.toFixed(2)}
                          </div>
                          <div style={{ fontSize: '12px', color: '#666' }}>Sentiment Score</div>
                        </div>
                        <div style={{ textAlign: 'center' }}>
                          <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#8b5cf6' }}>
                            {result.count || 0}
                          </div>
                          <div style={{ fontSize: '12px', color: '#666' }}>Mentions</div>
                        </div>
                        <div style={{ textAlign: 'center' }}>
                          <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#10b981' }}>
                            {result.category || 'N/A'}
                          </div>
                          <div style={{ fontSize: '12px', color: '#666' }}>Category</div>
                        </div>
                      </div>

                      {result.snippet && (
                        <div>
                          <div style={{ 
                            fontSize: '14px', 
                            fontWeight: '600', 
                            color: '#374151',
                            marginBottom: '10px'
                          }}>
                            💬 Sample Review:
                          </div>
                          <div style={{ 
                            fontSize: '13px', 
                            color: '#4b5563', 
                            padding: '12px', 
                            background: 'white', 
                            borderRadius: '8px',
                            border: '1px solid #e5e7eb',
                            borderLeft: `4px solid ${result.sentiment > 0.1 ? '#10b981' : result.sentiment < -0.1 ? '#ef4444' : '#6b7280'}`,
                            lineHeight: '1.5'
                          }}>
                            "{result.snippet}"
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {(!results.results || results.results.length === 0) && (
            <div className="card" style={{ textAlign: 'center', padding: '40px' }}>
              <h3 style={{ color: '#666' }}>No results found</h3>
              <p style={{ color: '#888' }}>Try a different search query or adjust your filters.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FeatureSearch;
