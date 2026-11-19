-- Deterministic sample data.
SELECT setseed(0.42);

INSERT INTO sales.customers (full_name, email, country, signup_date)
SELECT 'Customer ' || g, 'customer' || g || '@example.com',
       (ARRAY['Germany','India','USA','France','Brazil','Japan'])[1 + (g % 6)],
       date '2024-01-01' + (g * 3)
FROM generate_series(1, 120) g;

INSERT INTO sales.products (name, category, unit_price) VALUES
 ('Laptop Pro 14', 'Electronics', 1499.00), ('Wireless Mouse', 'Electronics', 29.90),
 ('Noise-cancelling Headphones', 'Electronics', 249.00), ('USB-C Hub', 'Electronics', 49.50),
 ('Espresso Machine', 'Home', 399.00), ('Chef Knife', 'Home', 89.00), ('Desk Lamp', 'Home', 39.99),
 ('Running Shoes', 'Sports', 119.00), ('Yoga Mat', 'Sports', 25.00), ('Water Bottle', 'Sports', 15.00),
 ('Data Science Handbook', 'Books', 45.00), ('SQL Cookbook', 'Books', 39.00);

INSERT INTO sales.orders (customer_id, order_date, status)
SELECT 1 + floor(random() * 120)::int,
       timestamptz '2025-01-01' + (random() * interval '365 days'),
       (ARRAY['pending','shipped','delivered','delivered','delivered','cancelled'])[1 + floor(random() * 6)::int]
FROM generate_series(1, 600);

INSERT INTO sales.order_items (order_id, product_id, quantity, unit_price)
SELECT DISTINCT ON (o.order_id, p.product_id) o.order_id, p.product_id, 1 + floor(random() * 3)::int, p.unit_price
FROM sales.orders o
CROSS JOIN LATERAL (SELECT product_id, unit_price FROM sales.products
                    ORDER BY random() + o.order_id * 0 LIMIT 1 + floor(random() * 3)::int) p;

ANALYZE;
