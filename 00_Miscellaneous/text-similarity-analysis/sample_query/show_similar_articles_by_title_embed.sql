#standardSQL
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


SELECT
  c.k1, c.k2,
  SUM(vv1*vv2) / (SQRT(SUM(POW(vv1,2))) * SQRT(SUM(POW(vv2,2)))) AS similarity
FROM
(
  SELECT
    a.key k1, a.val v1, b.key k2, b.val v2
  FROM
  (
    SELECT title key, title_embed val
    FROM reuters.embeddings
    WHERE title LIKE "%SECURITIES%"
    LIMIT 1
   ) a
  CROSS JOIN
  (
    SELECT title key, title_embed val
    FROM reuters.embeddings
  ) b
) c
, UNNEST(c.v1) vv1 with offset ind1 JOIN UNNEST(c.v2) vv2 with offset ind2 ON (ind1=ind2)
GROUP BY c.k1, c.k2
ORDER BY similarity DESC
