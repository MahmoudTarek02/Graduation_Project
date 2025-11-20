# 🧾 README — Internal Bakery Management System

## 📌 Project Overview

This system is an internal web platform for bakery staff and managers.
It acts as the frontend UI for the existing database and the CV team's product-detection system.

The platform centralizes:

- Sales tracking
- Inventory management
- Forecasting & analytics
- Alerts & notifications
- CV detection logs
- Supplier & customer management
- AI-powered insights and assistant

This is the main tool employees use to monitor bakery operations in real time.

## 📌 Problem Definition

The bakery currently struggles with:

- Manual stock tracking
- Human errors in counting
- Missing insights about trends
- Late detection of low stock or expired items
- No unified dashboard for staff
- No AI assistance for decisions

The goal is to build an internal website that solves all these problems by providing:

✔ Track sales, stock levels, and key statistics

✔ View forecasts and trends (sales, inventory, wastage)

✔ Manually adjust stock levels when CV detects wrong values

✔ Receive important alerts:

- Low inventory
- Items about to expire
- Camera issues (offline / blocked / low confidence)

## 📌 Database Structure

1. Food Items
- Name
- Pictures
- Count / Inventory Level
- Description
- Selling Price
- Cost Price
- Minimum Threshold
- Updated At
- Created At
- Expiration Date

2. Raw Material
- Name
- Count / Inventory
- Supplier
- Minimum Threshold
- Cost

3. Suppliers
- Supplier name
- Contact info
- Items supplied

4. Sales
- Product sold
- Quantity
- Price
- Timestamp
- Payment method
- Associated customer (optional)

5. Customers (Rewards Program)
- Name
- Contact information
- Reward points
- Purchase history

6. Operations Log
- User actions
- Inventory changes
- POS actions
- Adjustments
- Timestamp
- Reason/comment

7. CV Detection Logs (Extended Feature)
- Item detected
- Timestamp
- Confidence score
- Camera ID
- Snapshot path
- FP/FN tags (if reviewed)

8. Alerts
- Alert type
- Assigned to (Owner, Warehouse manager, Admin/IT)
- Timestamp
- Status (resolved/unresolved)

## 📌 Website Functionality

1. Authentication & Permissions
- Employee login with JWT
- Role-based access control:
  - Admin
  - Cashier
  - Warehouse manager
  - Viewer
- Optional PIN/QR login for fast access

2. Inventory Management
- Display all food items + raw materials
- Manual stock adjustments
- Override incorrect CV detections
- Track adjustment history in operations log
- View expiration dates
- Per-item history (trend & logs)

3. Alerts & Notifications

⚠ Low stock alerts

⚠ Expiring items

⚠ Camera alerts

- Offline
- Blocked
- Low confidence

Notifications are sent to:
- Owner
- Warehouse manager
- Admin / IT

In-app alert center + optional email/WhatsApp integration.

4. Point of Sale (POS)
- Process sales
- Deduct inventory automatically
- Add reward points
- Apply customer points
- Log full transaction info

5. Analytics & Trends
- Sales Analytics
  - Daily / weekly / monthly charts
  - Per-product sales performance
- Inventory Analytics
  - Stock consumption trends
  - Wastage (expired vs sold)
  - Restock frequency
- Forecasting (AI-powered)
  - Predict stock-out time
  - Predict demand for next days
  - Simple regression + moving average
- Camera Performance
  - Confidence metrics
  - Accuracy (FP/FN)
  - Per-camera stats
- AI Insights (From LLM)
  - Auto-generated weekly summary
  - "Most sold item this week…"
  - "Croissants likely to run out tomorrow…"

6. AI Assistant Bot (Full-Stack AI Feature)
- Suggest recipes
- Scale recipes based on inventory
- Answer logistics questions
- Query analytics (e.g., "Show stock trend for muffins last week")
- Access internal knowledge (RAG):
  - Manuals
  - Recipes
  - SOPs
  - Logs

## 📌 Tech Stack

### 🎨 Frontend
- React
  - Build all website pages
  - Render dashboards, trends, inventory, alerts
  - Handle user interactions
- CSS (Plain or Tailwind)
  - Clean styling
  - Component-based UI
- Axios
  - Communicate with backend API
  - Handle authentication tokens
- Charts (Recharts / Chart.js)
  - Display analytics, trends, forecasts

### ⚙️ Backend
- Node.js + Express
  - Handles all backend logic:
    - Authentication (JWT)
    - Getting products & raw materials
    - Updating inventory
    - Processing sales
    - Creating alerts
    - Logging operations
    - CV integration API
    - AI assistant API
    - Forecasting logic
- PostgreSQL
  - Stores:
    - Users
    - Products
    - Raw materials
    - Suppliers
    - Sales
    - Alerts
    - Customers
    - Logs
    - Camera detections
- JWT Authentication
  - Secure login
  - Role-based access

## 🎯 Absolute Minimum Viable System
- Frontend
  - React + CSS
- Backend
  - Node.js
  - Express
  - PostgreSQL
