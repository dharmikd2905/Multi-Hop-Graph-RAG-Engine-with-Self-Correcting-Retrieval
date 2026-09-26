# Internal Platform Architecture Notes

## Component A: API Gateway
The API Gateway (Component A) is the single entry point for all external
traffic. It performs authentication and rate limiting before forwarding
requests. Component A depends on the Auth Service for token validation.

## Auth Service
The Auth Service issues and validates JWT tokens. It depends on the User
Database for credential lookups. If the Auth Service is unavailable, the
API Gateway falls back to cached tokens for up to 5 minutes.

## Component B: Message Queue
The Message Queue (Component B) receives events forwarded by the Auth
Service after a successful login, and publishes a "user_logged_in" event.
Component B is built on Kafka and buffers events for downstream consumers.

## Component C: Analytics Pipeline
The Analytics Pipeline (Component C) subscribes to the Message Queue and
consumes "user_logged_in" events to update real-time dashboards. Component
C depends on the Message Queue for its event stream and writes aggregated
metrics to the Data Warehouse.

## Data Warehouse
The Data Warehouse stores aggregated metrics produced by the Analytics
Pipeline. It is queried by the Reporting Service to generate weekly usage
reports for stakeholders.

## Summary of Indirect Effects
Because the API Gateway (Component A) triggers login events that flow
through the Auth Service into the Message Queue (Component B), and the
Analytics Pipeline (Component C) consumes those same events from the
Message Queue, any change in Component A's authentication latency
indirectly affects the freshness of metrics produced by Component C --
even though Component A and Component C never communicate directly.
