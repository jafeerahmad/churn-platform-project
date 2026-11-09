-- MySQL 8 analytics queries (tables: customers, events).
-- Event activity and order values are simulated in this demonstration project.

-- 1. Daily active users (DAU): customers with at least one login per day.
SELECT DATE(event_time) AS day,
       COUNT(DISTINCT customer_id) AS daily_active_users
FROM events
WHERE event_type = 'login'
GROUP BY DATE(event_time)
ORDER BY day;

-- 2. Daily simulated order value and average order value.
SELECT DATE(event_time) AS day,
       ROUND(SUM(amount), 2) AS simulated_order_value,
       ROUND(AVG(amount), 2) AS average_order_value,
       COUNT(*) AS orders
FROM events
WHERE event_type = 'order'
GROUP BY DATE(event_time)
ORDER BY day;

-- 3. Churn rate by contract type. Churn is stored as 1 (Yes) / 0 (No).
SELECT Contract,
       COUNT(*) AS customers,
       ROUND(AVG(Churn) * 100, 1) AS churn_pct
FROM customers
GROUP BY Contract
ORDER BY churn_pct DESC;

-- 4. Top 10 customers by simulated order value in the last 30 days.
SELECT customer_id,
       ROUND(SUM(amount), 2) AS simulated_order_value
FROM events
WHERE event_type = 'order'
  AND event_time >= NOW() - INTERVAL 30 DAY
GROUP BY customer_id
ORDER BY simulated_order_value DESC
LIMIT 10;

-- 5. Customers with at least three simulated support tickets in the last 7 days.
SELECT customer_id,
       COUNT(*) AS support_tickets_7d
FROM events
WHERE event_type = 'support_ticket'
  AND event_time >= NOW() - INTERVAL 7 DAY
GROUP BY customer_id
HAVING COUNT(*) >= 3
ORDER BY support_tickets_7d DESC;

-- 6. Weekly login retention. The retention denominator is the prior week's active
-- users. Week-start dates avoid incorrect YEARWEEK arithmetic across New Year.
WITH weekly_activity AS (
    SELECT DISTINCT
           customer_id,
           DATE_SUB(DATE(event_time), INTERVAL WEEKDAY(event_time) DAY) AS week_start
    FROM events
    WHERE event_type = 'login'
),
weekly_counts AS (
    SELECT week_start, COUNT(*) AS active_users
    FROM weekly_activity
    GROUP BY week_start
),
weekly_retained AS (
    SELECT current_week.week_start,
           COUNT(*) AS retained_users
    FROM weekly_activity AS current_week
    INNER JOIN weekly_activity AS previous_week
        ON previous_week.customer_id = current_week.customer_id
       AND previous_week.week_start = DATE_SUB(current_week.week_start, INTERVAL 7 DAY)
    GROUP BY current_week.week_start
)
SELECT current_counts.week_start,
       previous_counts.active_users AS previous_week_active_users,
       current_counts.active_users AS current_week_active_users,
       COALESCE(weekly_retained.retained_users, 0) AS retained_users,
       ROUND(
           100.0 * COALESCE(weekly_retained.retained_users, 0)
           / NULLIF(previous_counts.active_users, 0),
           1
       ) AS retention_pct
FROM weekly_counts AS current_counts
LEFT JOIN weekly_counts AS previous_counts
    ON previous_counts.week_start = DATE_SUB(current_counts.week_start, INTERVAL 7 DAY)
LEFT JOIN weekly_retained
    ON weekly_retained.week_start = current_counts.week_start
ORDER BY current_counts.week_start;
