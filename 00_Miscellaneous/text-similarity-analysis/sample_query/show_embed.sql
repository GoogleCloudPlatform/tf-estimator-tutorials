#standardSQL

SELECT
  title,
  content,
  title_embed,
  content_embed
FROM
  `reuters.embeddings`
LIMIT 1
