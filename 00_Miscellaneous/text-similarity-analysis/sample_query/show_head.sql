#standardSQL

SELECT
  topics,
  title,
  content
FROM
  `reuters.embeddings`
WHERE topics like '%trade%'
ORDER BY LENGTH(title) desc
LIMIT 5
