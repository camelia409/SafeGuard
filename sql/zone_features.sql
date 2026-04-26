-- Zone DNA feature queries
-- Source table: restaurants_zoned

-- 1) Demand density: total restaurants available in each zone.
SELECT
    zone,
    COUNT(*) AS restaurant_count
FROM restaurants_zoned
GROUP BY zone;

-- 2) Delivery engagement: average votes among online-ordering restaurants.
SELECT
    zone,
    CAST(SUM(votes) AS FLOAT) / COUNT(*) AS engagement_score
FROM restaurants_zoned
WHERE online_order = 1
GROUP BY zone;

-- 3) Demand volatility proxy: share of Quick Bites restaurants in each zone.
SELECT
    zone,
    CAST(SUM(CASE WHEN rest_type = 'Quick Bites' THEN 1 ELSE 0 END) AS FLOAT)
        / COUNT(*) AS volatility_index
FROM restaurants_zoned
GROUP BY zone;

-- 4) Spending proxy: average cost for two where price data is present.
SELECT
    zone,
    AVG(approx_cost) AS affluence_proxy
FROM restaurants_zoned
WHERE approx_cost IS NOT NULL
GROUP BY zone;

-- 5) Digital readiness: share of restaurants accepting online orders.
SELECT
    zone,
    CAST(SUM(online_order) AS FLOAT) / COUNT(*) AS online_penetration
FROM restaurants_zoned
GROUP BY zone;

