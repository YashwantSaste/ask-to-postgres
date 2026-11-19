-- Sample e-commerce schema used for the demo and the integration tests.
CREATE SCHEMA sales;
COMMENT ON SCHEMA sales IS 'Orders, customers and products of the web shop';

CREATE TABLE sales.customers (
    customer_id  serial PRIMARY KEY,
    full_name    text NOT NULL,
    email        text UNIQUE NOT NULL,
    country      text NOT NULL,
    signup_date  date NOT NULL
);
COMMENT ON TABLE sales.customers IS 'Registered shop customers';

CREATE TABLE sales.products (
    product_id  serial PRIMARY KEY,
    name        text NOT NULL,
    category    text NOT NULL,
    unit_price  numeric(10,2) NOT NULL CHECK (unit_price >= 0)
);

CREATE TABLE sales.orders (
    order_id     serial PRIMARY KEY,
    customer_id  int NOT NULL REFERENCES sales.customers,
    order_date   timestamptz NOT NULL,
    status       text NOT NULL CHECK (status IN ('pending','shipped','delivered','cancelled'))
);
COMMENT ON COLUMN sales.orders.status IS 'Lifecycle state: pending, shipped, delivered or cancelled';

CREATE TABLE sales.order_items (
    order_id    int REFERENCES sales.orders,
    product_id  int REFERENCES sales.products,
    quantity    int NOT NULL CHECK (quantity > 0),
    unit_price  numeric(10,2) NOT NULL,
    PRIMARY KEY (order_id, product_id)
);
COMMENT ON COLUMN sales.order_items.unit_price IS 'Price actually charged per unit (revenue = quantity * unit_price)';

CREATE VIEW sales.order_totals AS
SELECT o.order_id, o.customer_id, o.order_date, o.status, sum(i.quantity * i.unit_price) AS total
FROM sales.orders o JOIN sales.order_items i USING (order_id)
GROUP BY o.order_id;
