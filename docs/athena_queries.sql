-- Athena Queries for Sentiment-Driven Product Feature Insights

-- Create external table for raw reviews
CREATE EXTERNAL TABLE IF NOT EXISTS amazon_reviews_all_beauty (
  rating double,
  title string,
  text string,
  asin string,
  parent_asin string,
  timestamp bigint,
  user_id string,
  verified_purchase boolean
)
PARTITIONED BY (
  year string,
  month string,
  day string
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://your-bucket-name/raw/All_Beauty/'
TBLPROPERTIES (
  'classification' = 'json',
  'typeOfData' = 'file'
);

-- Create external table for sentiment insights
CREATE EXTERNAL TABLE IF NOT EXISTS product_sentiment_insights (
  parent_asin string,
  feature string,
  agg_score_sum double,
  agg_score_count int,
  positive_snippets array<string>,
  negative_snippets array<string>,
  last_updated bigint,
  category string
)
PARTITIONED BY (
  year string,
  month string,
  day string
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://your-bucket-name/processed/sentiment/'
TBLPROPERTIES (
  'classification' = 'json',
  'typeOfData' = 'file'
);

-- Sample queries

-- 1. Get total number of reviews
SELECT COUNT(*) as total_reviews
FROM amazon_reviews_all_beauty;

-- 2. Get reviews for a specific product
SELECT 
  asin,
  parent_asin,
  rating,
  title,
  text,
  from_unixtime(timestamp/1000) as review_date
FROM amazon_reviews_all_beauty
WHERE parent_asin = 'B00YQ6X8EO'
ORDER BY timestamp DESC
LIMIT 10;

-- 3. Get average rating by product
SELECT 
  parent_asin,
  COUNT(*) as review_count,
  AVG(rating) as avg_rating,
  MIN(rating) as min_rating,
  MAX(rating) as max_rating
FROM amazon_reviews_all_beauty
GROUP BY parent_asin
HAVING COUNT(*) >= 10
ORDER BY avg_rating DESC
LIMIT 20;

-- 4. Get sentiment insights for a specific product
SELECT 
  parent_asin,
  feature,
  agg_score_sum / agg_score_count as avg_sentiment,
  agg_score_count as mention_count,
  size(positive_snippets) as positive_snippets_count,
  size(negative_snippets) as negative_snippets_count,
  from_unixtime(last_updated/1000) as last_updated
FROM product_sentiment_insights
WHERE parent_asin = 'B00YQ6X8EO'
ORDER BY avg_sentiment DESC;

-- 5. Get top features by sentiment across all products
SELECT 
  feature,
  COUNT(DISTINCT parent_asin) as product_count,
  AVG(agg_score_sum / agg_score_count) as avg_sentiment,
  SUM(agg_score_count) as total_mentions
FROM product_sentiment_insights
WHERE agg_score_count > 0
GROUP BY feature
HAVING COUNT(DISTINCT parent_asin) >= 5
ORDER BY avg_sentiment DESC
LIMIT 20;

-- 6. Get products with highest sentiment for a specific feature
SELECT 
  parent_asin,
  feature,
  agg_score_sum / agg_score_count as avg_sentiment,
  agg_score_count as mention_count
FROM product_sentiment_insights
WHERE feature = 'battery_life'
  AND agg_score_count >= 5
ORDER BY avg_sentiment DESC
LIMIT 10;

-- 7. Get sentiment trends over time
SELECT 
  DATE(from_unixtime(timestamp/1000)) as review_date,
  COUNT(*) as review_count,
  AVG(rating) as avg_rating
FROM amazon_reviews_all_beauty
WHERE parent_asin = 'B00YQ6X8EO'
  AND timestamp >= unix_timestamp('2024-01-01') * 1000
GROUP BY DATE(from_unixtime(timestamp/1000))
ORDER BY review_date;

-- 8. Get reviews with specific sentiment patterns
SELECT 
  asin,
  parent_asin,
  rating,
  title,
  text,
  from_unixtime(timestamp/1000) as review_date
FROM amazon_reviews_all_beauty
WHERE parent_asin = 'B00YQ6X8EO'
  AND (
    LOWER(text) LIKE '%battery%' OR
    LOWER(text) LIKE '%camera%' OR
    LOWER(text) LIKE '%screen%'
  )
ORDER BY timestamp DESC
LIMIT 20;

-- 9. Get sentiment distribution by rating
SELECT 
  rating,
  COUNT(*) as review_count,
  AVG(agg_score_sum / agg_score_count) as avg_sentiment
FROM amazon_reviews_all_beauty a
LEFT JOIN product_sentiment_insights p ON a.parent_asin = p.parent_asin
WHERE p.agg_score_count > 0
GROUP BY rating
ORDER BY rating;

-- 10. Get products with most negative sentiment
SELECT 
  parent_asin,
  feature,
  agg_score_sum / agg_score_count as avg_sentiment,
  agg_score_count as mention_count,
  negative_snippets
FROM product_sentiment_insights
WHERE agg_score_count >= 10
  AND agg_score_sum / agg_score_count < -0.5
ORDER BY avg_sentiment ASC
LIMIT 20;

-- 11. Get sentiment insights by category
SELECT 
  category,
  COUNT(DISTINCT parent_asin) as product_count,
  COUNT(DISTINCT feature) as feature_count,
  AVG(agg_score_sum / agg_score_count) as avg_sentiment
FROM product_sentiment_insights
WHERE agg_score_count > 0
GROUP BY category
ORDER BY avg_sentiment DESC;

-- 12. Get feature popularity across products
SELECT 
  feature,
  COUNT(DISTINCT parent_asin) as product_count,
  SUM(agg_score_count) as total_mentions,
  AVG(agg_score_sum / agg_score_count) as avg_sentiment
FROM product_sentiment_insights
WHERE agg_score_count > 0
GROUP BY feature
HAVING COUNT(DISTINCT parent_asin) >= 3
ORDER BY total_mentions DESC
LIMIT 30;

-- 13. Get recent sentiment insights (last 30 days)
SELECT 
  parent_asin,
  feature,
  agg_score_sum / agg_score_count as avg_sentiment,
  agg_score_count as mention_count,
  from_unixtime(last_updated/1000) as last_updated
FROM product_sentiment_insights
WHERE last_updated >= unix_timestamp(date_add('day', -30, current_date)) * 1000
ORDER BY last_updated DESC
LIMIT 50;

-- 14. Get sentiment correlation with ratings
SELECT 
  a.parent_asin,
  AVG(a.rating) as avg_rating,
  AVG(p.agg_score_sum / p.agg_score_count) as avg_sentiment,
  COUNT(*) as review_count
FROM amazon_reviews_all_beauty a
JOIN product_sentiment_insights p ON a.parent_asin = p.parent_asin
WHERE p.agg_score_count > 0
GROUP BY a.parent_asin
HAVING COUNT(*) >= 10
ORDER BY avg_rating DESC, avg_sentiment DESC
LIMIT 20;

-- 15. Get sentiment insights with positive and negative snippets
SELECT 
  parent_asin,
  feature,
  agg_score_sum / agg_score_count as avg_sentiment,
  agg_score_count as mention_count,
  positive_snippets[1] as top_positive_snippet,
  negative_snippets[1] as top_negative_snippet
FROM product_sentiment_insights
WHERE parent_asin = 'B00YQ6X8EO'
  AND agg_score_count >= 5
ORDER BY avg_sentiment DESC;
