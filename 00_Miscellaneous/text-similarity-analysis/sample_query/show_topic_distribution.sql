#standardSQL

SELECT
  topic,
  count(title) title_count
FROM
  `reuters.embeddings` t,
  UNNEST(SPLIT(t.topics, ',')) topic
GROUP BY topic
ORDER BY title_count DESC
