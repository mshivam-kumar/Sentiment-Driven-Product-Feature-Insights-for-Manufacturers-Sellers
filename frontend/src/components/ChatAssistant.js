import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { API_ENDPOINTS } from '../config/api';

const ChatAssistant = () => {
  const [question, setQuestion] = useState('');
  const [useTransformer, setUseTransformer] = useState(true);
  const [loading, setLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState(() => {
    // Load from localStorage on component mount
    const saved = localStorage.getItem('chatHistory');
    return saved ? JSON.parse(saved) : [];
  });
  const [error, setError] = useState(null);
  const [chatStatus, setChatStatus] = useState(null);
  const [expandedReviews, setExpandedReviews] = useState(() => {
    // Load from localStorage on component mount
    const saved = localStorage.getItem('expandedReviews');
    return saved ? JSON.parse(saved) : {};
  });

  // Example questions for quick testing
  const exampleQuestions = [
    {
      category: "Product Quality",
      questions: [
        "What do customers say about product quality?",
        "How do customers rate the build quality?",
        "What are the main quality issues mentioned?"
      ]
    },
    {
      category: "Design & Features",
      questions: [
        "How do customers feel about the design?",
        "What features do customers like most?",
        "What design improvements do customers suggest?"
      ]
    },
    {
      category: "Performance",
      questions: [
        "How does the product perform according to customers?",
        "What performance issues are mentioned?",
        "How satisfied are customers with performance?"
      ]
    },
    {
      category: "Value & Price",
      questions: [
        "Is the product worth the price according to customers?",
        "How do customers feel about the value for money?",
        "What do customers say about pricing?"
      ]
    }
  ];

  useEffect(() => {
    // Check chat status on component mount
    checkChatStatus();
  }, []);

  // Save chat history to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
  }, [chatHistory]);

  // Save expanded reviews to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('expandedReviews', JSON.stringify(expandedReviews));
  }, [expandedReviews]);

  const checkChatStatus = async () => {
    try {
      const response = await fetch(`${API_ENDPOINTS.CHAT_ASSISTANT}/status`);
      const data = await response.json();
      setChatStatus(data);
    } catch (err) {
      console.error('Failed to check chat status:', err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_ENDPOINTS.CHAT_ASSISTANT}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
          use_transformer: useTransformer,
          session_id: 'default'
        })
      });

      const data = await response.json();

      if (data.success) {
        const newChat = {
          question: question,
          answer: data.answer,
          generation_method: data.generation_method,
          is_fine_tuned: data.is_fine_tuned,
          model_type: data.model_type,
          supporting_reviews: data.supporting_reviews || [],
          timestamp: new Date().toLocaleTimeString()
        };

        setChatHistory(prev => [...prev, newChat]);
        setQuestion(''); // Clear input after successful submission
      } else {
        setError(data.detail || 'Failed to get response from AI assistant');
      }
    } catch (err) {
      setError('Network error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (exampleQuestion) => {
    setQuestion(exampleQuestion);
  };

  const clearChatHistory = () => {
    setChatHistory([]);
    setExpandedReviews({});
  };

  const toggleExpandedReviews = (chatIndex) => {
    setExpandedReviews(prev => ({
      ...prev,
      [chatIndex]: !prev[chatIndex]
    }));
  };

  const getMethodEmoji = (chat) => {
    if (chat.is_fine_tuned) return '🎯';
    if (chat.generation_method === 'transformer') return '🤖';
    return '📝';
  };

  const getMethodText = (chat) => {
    if (chat.is_fine_tuned) return 'Fine-tuned TinyLlama';
    if (chat.generation_method === 'transformer') return 'Pre-trained TinyLlama';
    return 'Rule-based';
  };

  const generateChartsData = (supportingReviews) => {
    if (!supportingReviews || supportingReviews.length === 0) return null;

    // Sentiment distribution data
    const sentimentData = [
      { name: 'Positive', value: supportingReviews.filter(r => r.sentiment > 0.1).length, color: '#10b981' },
      { name: 'Negative', value: supportingReviews.filter(r => r.sentiment < -0.1).length, color: '#ef4444' },
      { name: 'Neutral', value: supportingReviews.filter(r => r.sentiment >= -0.1 && r.sentiment <= 0.1).length, color: '#6b7280' }
    ];

    // ASIN sentiment scores
    const asinData = supportingReviews.map((review, index) => ({
      asin: review.asin || `Review ${index + 1}`,
      sentiment: review.sentiment || 0,
      rating: review.rating || 0,
      relevance: review.relevance_score || 0
    }));

    return { sentimentData, asinData };
  };

  return (
    <div>
      <h2>🤖 AI Chat Assistant</h2>
      <p>Ask me anything about products and customer sentiment using natural language!</p>

      {chatStatus && (
        <div className="card" style={{ 
          background: chatStatus.available ? '#f0f9ff' : '#fef2f2',
          border: `1px solid ${chatStatus.available ? '#10b981' : '#ef4444'}`
        }}>
          <h4 style={{ 
            color: chatStatus.available ? '#10b981' : '#ef4444',
            margin: '0 0 10px 0'
          }}>
            {chatStatus.available ? '✅ AI Assistant Ready' : '❌ AI Assistant Unavailable'}
          </h4>
          <p style={{ margin: '0 0 10px 0', color: '#666' }}>{chatStatus.status}</p>
          {chatStatus.features && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '10px' }}>
              <div>🤖 Transformer: {chatStatus.features.transformer_generation ? '✅' : '❌'}</div>
              <div>🔍 Semantic Search: {chatStatus.features.semantic_search ? '✅' : '❌'}</div>
              <div>🎯 Fine-tuned: {chatStatus.features.fine_tuned_model ? '✅' : '⚠️'}</div>
            </div>
          )}
        </div>
      )}

      <div className="card">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="question">Ask your question:</label>
            <textarea
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g., What do customers say about the battery life?"
              rows="3"
              required
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '20px', marginBottom: '20px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={useTransformer}
                onChange={(e) => setUseTransformer(e.target.checked)}
              />
              <span>Use AI Model (Transformer-based generation)</span>
            </label>
          </div>

          <button type="submit" className="btn" disabled={loading}>
            {loading ? 'AI is thinking...' : '💬 Ask AI'}
          </button>
        </form>

        <div style={{ marginTop: '20px' }}>
          <h4>💡 Example Questions:</h4>
          {exampleQuestions.map((category, categoryIndex) => (
            <div key={categoryIndex} style={{ marginBottom: '15px' }}>
              <h5 style={{ margin: '0 0 8px 0', color: '#333' }}>{category.category}:</h5>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {category.questions.map((exampleQuestion, questionIndex) => (
                  <button
                    key={questionIndex}
                    className="btn btn-secondary"
                    onClick={() => handleExampleClick(exampleQuestion)}
                    style={{ fontSize: '12px', padding: '6px 12px' }}
                  >
                    {exampleQuestion}
                  </button>
                ))}
              </div>
            </div>
          ))}
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
          <p>AI is thinking...</p>
        </div>
      )}

      {chatHistory.length > 0 && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h3>💬 Chat History</h3>
            <button 
              className="btn btn-secondary" 
              onClick={clearChatHistory}
              style={{ fontSize: '14px', padding: '8px 16px' }}
            >
              🗑️ Clear History
            </button>
          </div>
          
          <div style={{ display: 'grid', gap: '20px' }}>
            {chatHistory.slice().reverse().map((chat, index) => (
              <div key={index} style={{ 
                padding: '20px', 
                border: '1px solid #e1e5e9', 
                borderRadius: '12px',
                background: '#f9fafb'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                  <h4 style={{ margin: 0, color: '#333' }}>
                    {getMethodEmoji(chat)} Q: {chat.question}
                  </h4>
                  <span style={{ fontSize: '12px', color: '#666' }}>{chat.timestamp}</span>
                </div>
                
                <div style={{ marginBottom: '15px' }}>
                  <strong>AI Response:</strong>
                  <p style={{ 
                    margin: '8px 0', 
                    padding: '12px', 
                    background: 'white', 
                    borderRadius: '8px',
                    border: '1px solid #e1e5e9'
                  }}>
                    {chat.answer}
                  </p>
                </div>

                <div style={{ 
                  fontSize: '12px', 
                  color: '#666', 
                  marginBottom: '15px',
                  padding: '8px 12px',
                  background: '#f3f4f6',
                  borderRadius: '6px'
                }}>
                  Generated using: {getMethodText(chat)} ({chat.model_type})
                </div>

                {chat.supporting_reviews && chat.supporting_reviews.length > 0 && (
                  <div>
                    <strong>📊 Supporting Evidence & Analysis:</strong>
                    
                    {/* Charts Section */}
                    {(() => {
                      const chartsData = generateChartsData(chat.supporting_reviews);
                      if (chartsData) {
                        return (
                          <div style={{ marginTop: '15px', marginBottom: '20px' }}>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '20px' }}>
                              {/* Sentiment Distribution Chart */}
                              <div>
                                <h5 style={{ textAlign: 'center', marginBottom: '10px' }}>Sentiment Distribution</h5>
                                <ResponsiveContainer width="100%" height={250}>
                                  <PieChart>
                                    <Pie
                                      data={chartsData.sentimentData}
                                      cx="50%"
                                      cy="50%"
                                      labelLine={false}
                                      label={({ name, value, percent }) => {
                                        if (value === 0) return '';
                                        return `${name}: ${value}`;
                                      }}
                                      outerRadius={80}
                                      fill="#8884d8"
                                      dataKey="value"
                                    >
                                      {chartsData.sentimentData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                      ))}
                                    </Pie>
                                    <Tooltip 
                                      formatter={(value, name) => [value, name]}
                                      labelFormatter={(label) => `Sentiment: ${label}`}
                                    />
                                  </PieChart>
                                </ResponsiveContainer>
                              </div>

                              {/* ASIN Sentiment Chart */}
                              <div>
                                <h5 style={{ textAlign: 'center', marginBottom: '10px' }}>Sentiment by Product</h5>
                                <ResponsiveContainer width="100%" height={250}>
                                  <BarChart data={chartsData.asinData} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis 
                                      dataKey="asin" 
                                      angle={-45} 
                                      textAnchor="end" 
                                      height={100}
                                      fontSize={12}
                                    />
                                    <YAxis />
                                    <Tooltip 
                                      formatter={(value, name) => [value, name === 'sentiment' ? 'Sentiment Score' : name === 'rating' ? 'Rating' : 'Relevance']}
                                      labelFormatter={(label) => `Product: ${label}`}
                                    />
                                    <Bar dataKey="sentiment" fill="#3b82f6" />
                                  </BarChart>
                                </ResponsiveContainer>
                              </div>
                            </div>
                          </div>
                        );
                      }
                      return null;
                    })()}

                    {/* Supporting Reviews - Collapsible */}
                    <div style={{ marginTop: '20px' }}>
                      <button
                        onClick={() => toggleExpandedReviews(index)}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '8px',
                          padding: '12px 16px',
                          background: '#f8fafc',
                          border: '1px solid #e5e7eb',
                          borderRadius: '8px',
                          cursor: 'pointer',
                          width: '100%',
                          fontSize: '16px',
                          fontWeight: '600',
                          color: '#374151',
                          transition: 'all 0.2s ease'
                        }}
                        onMouseEnter={(e) => {
                          e.target.style.background = '#f1f5f9';
                        }}
                        onMouseLeave={(e) => {
                          e.target.style.background = '#f8fafc';
                        }}
                      >
                        <span>📝 Supporting Reviews ({chat.supporting_reviews.length})</span>
                        <span style={{ 
                          transform: expandedReviews[index] ? 'rotate(180deg)' : 'rotate(0deg)',
                          transition: 'transform 0.2s ease',
                          fontSize: '14px'
                        }}>
                          ▼
                        </span>
                      </button>
                      
                      {expandedReviews[index] && (
                        <div style={{ 
                          marginTop: '15px',
                          padding: '20px',
                          background: '#f9fafb',
                          borderRadius: '8px',
                          border: '1px solid #e5e7eb'
                        }}>
                          <div style={{ display: 'grid', gap: '15px' }}>
                            {chat.supporting_reviews.slice(0, 3).map((review, reviewIndex) => (
                              <div key={reviewIndex} style={{ 
                                padding: '15px', 
                                background: 'white', 
                                borderRadius: '8px',
                                border: '1px solid #e1e5e9',
                                boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
                              }}>
                                <p style={{ 
                                  margin: '0 0 12px 0', 
                                  fontSize: '14px',
                                  lineHeight: '1.5',
                                  color: '#374151'
                                }}>
                                  <strong>Review {reviewIndex + 1}:</strong> {review.text}
                                </p>
                                
                                <div style={{ 
                                  display: 'grid', 
                                  gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', 
                                  gap: '12px', 
                                  fontSize: '13px',
                                  padding: '10px',
                                  background: '#f8fafc',
                                  borderRadius: '6px',
                                  border: '1px solid #e5e7eb'
                                }}>
                                  <div style={{ textAlign: 'center' }}>
                                    <div style={{ fontWeight: 'bold', color: '#374151' }}>Sentiment</div>
                                    <div style={{ 
                                      fontSize: '16px', 
                                      fontWeight: 'bold',
                                      color: review.sentiment > 0.1 ? '#10b981' : review.sentiment < -0.1 ? '#ef4444' : '#6b7280'
                                    }}>
                                      {review.sentiment?.toFixed(2) || 'N/A'}
                                    </div>
                                  </div>
                                  <div style={{ textAlign: 'center' }}>
                                    <div style={{ fontWeight: 'bold', color: '#374151' }}>Rating</div>
                                    <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#3b82f6' }}>
                                      {review.rating || 'N/A'}/5
                                    </div>
                                  </div>
                                  <div style={{ textAlign: 'center' }}>
                                    <div style={{ fontWeight: 'bold', color: '#374151' }}>Relevance</div>
                                    <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#8b5cf6' }}>
                                      {review.relevance_score?.toFixed(2) || 'N/A'}
                                    </div>
                                  </div>
                                  <div style={{ textAlign: 'center' }}>
                                    <div style={{ fontWeight: 'bold', color: '#374151' }}>ASIN</div>
                                    <div style={{ 
                                      fontSize: '12px', 
                                      fontWeight: 'bold', 
                                      color: '#6b7280',
                                      fontFamily: 'monospace',
                                      background: '#f3f4f6',
                                      padding: '4px 8px',
                                      borderRadius: '4px'
                                    }}>
                                      {review.asin || 'Unknown'}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatAssistant;
