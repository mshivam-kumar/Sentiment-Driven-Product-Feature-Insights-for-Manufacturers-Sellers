import React, { useState } from 'react';

const TestProductAnalysis = () => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const testAPI = async () => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      console.log('Testing API call...');
      const response = await fetch('http://localhost:8001/api/v1/product/B08JTNQFZY');
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Raw API Response:', data);

      if (data.success) {
        // Transform the data to match expected format
        const features = data.data.features || {};
        console.log('Features object:', features);
        console.log('Features keys:', Object.keys(features));
        console.log('Features length:', Object.keys(features).length);
        
        const featuresList = Object.entries(features).map(([featureName, featureData]) => {
          console.log(`Processing feature: ${featureName}`, featureData);
          return {
            feature: featureName,
            sentiment: featureData.score || 0,
            count: featureData.count || 0,
            snippets: [...(featureData.positive_snippets || []), ...(featureData.negative_snippets || [])].slice(0, 3)
          };
        });
        
        const transformedData = {
          asin: data.data.asin,
          overall_sentiment: data.data.overall_sentiment,
          total_reviews: data.data.total_reviews,
          features: featuresList
        };
        
        console.log('Transformed data:', transformedData);
        console.log('Features count:', transformedData.features.length);
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

  return (
    <div style={{ padding: '20px', border: '1px solid #ccc', margin: '20px' }}>
      <h2>🧪 Test Product Analysis</h2>
      <button onClick={testAPI} disabled={loading}>
        {loading ? 'Testing...' : 'Test API Call'}
      </button>
      
      {error && (
        <div style={{ color: 'red', margin: '10px 0' }}>
          Error: {error}
        </div>
      )}
      
      {results && (
        <div style={{ margin: '20px 0' }}>
          <h3>Results:</h3>
          <p><strong>ASIN:</strong> {results.asin}</p>
          <p><strong>Overall Sentiment:</strong> {results.overall_sentiment}</p>
          <p><strong>Total Reviews:</strong> {results.total_reviews}</p>
          <p><strong>Features Analyzed:</strong> {results.features?.length || 0}</p>
          
          {results.features && results.features.length > 0 && (
            <div>
              <h4>Features:</h4>
              {results.features.map((feature, index) => (
                <div key={index} style={{ margin: '10px 0', padding: '10px', background: '#f5f5f5' }}>
                  <strong>{feature.feature}:</strong> {feature.sentiment} ({feature.count} reviews)
                  {feature.snippets.length > 0 && (
                    <div style={{ fontSize: '12px', color: '#666' }}>
                      "{feature.snippets[0]?.substring(0, 100)}..."
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TestProductAnalysis;



