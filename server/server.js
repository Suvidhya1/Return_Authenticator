/**
 * server.js
 * ---------
 * Express API that sits between the React frontend and the Python ML service.
 *
 * Responsibilities:
 *   - Receive a return request from the frontend
 *   - Forward the features to the Python /predict service
 *   - Save the request + prediction to MongoDB (so agents can review history)
 *   - Return the prediction to the frontend
 *
 * MongoDB is optional in dev: if no MONGODB_URI is set, the server still works
 * and just skips persistence (handy for quick local testing).
 */

const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 5000;
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://localhost:5001";
const MONGODB_URI = process.env.MONGODB_URI; // optional

// --- MongoDB model -------------------------------------------------------
const returnSchema = new mongoose.Schema(
  {
    orderValue: Number,
    productCategory: String,
    customerReturnRate: Number,
    daysSincePurchase: Number,
    priorOrders: Number,
    returnReason: String,
    isHighValue: Number,
    label: String,
    fraudScore: Number,
    recommendedAction: String,
  },
  { timestamps: true }
);
const ReturnRequest = mongoose.model("ReturnRequest", returnSchema);

let dbReady = false;
if (MONGODB_URI) {
  mongoose
    .connect(MONGODB_URI)
    .then(() => {
      dbReady = true;
      console.log("Connected to MongoDB");
    })
    .catch((err) => console.error("MongoDB connection error:", err.message));
} else {
  console.log("No MONGODB_URI set -- running without persistence (dev mode).");
}

// --- Routes --------------------------------------------------------------
app.get("/api/health", (_req, res) => res.json({ status: "ok", dbReady }));

/**
 * POST /api/returns
 * Body: the raw return-request features (camelCase from the frontend).
 * Calls the ML service, persists the result, returns the prediction.
 */
app.post("/api/returns", async (req, res) => {
  try {
    const b = req.body;

    // Map camelCase (frontend convention) -> snake_case (Python convention).
    const mlPayload = {
      order_value: Number(b.orderValue),
      product_category: b.productCategory,
      customer_return_rate: Number(b.customerReturnRate),
      days_since_purchase: Number(b.daysSincePurchase),
      prior_orders: Number(b.priorOrders),
      return_reason: b.returnReason,
      is_high_value: Number(b.isHighValue),
    };

    const mlRes = await fetch(`${ML_SERVICE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(mlPayload),
    });

    if (!mlRes.ok) {
      const text = await mlRes.text();
      return res.status(502).json({ error: "ML service error", detail: text });
    }

    const prediction = await mlRes.json();

    if (dbReady) {
      await ReturnRequest.create({
        orderValue: mlPayload.order_value,
        productCategory: mlPayload.product_category,
        customerReturnRate: mlPayload.customer_return_rate,
        daysSincePurchase: mlPayload.days_since_purchase,
        priorOrders: mlPayload.prior_orders,
        returnReason: mlPayload.return_reason,
        isHighValue: mlPayload.is_high_value,
        label: prediction.label,
        fraudScore: prediction.fraud_score,
        recommendedAction: prediction.recommended_action,
      });
    }

    res.json(prediction);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "internal error", detail: err.message });
  }
});

/**
 * GET /api/returns
 * Returns recent processed requests for the agent dashboard.
 */
app.get("/api/returns", async (_req, res) => {
  if (!dbReady) return res.json([]);
  const items = await ReturnRequest.find().sort({ createdAt: -1 }).limit(50);
  res.json(items);
});

app.listen(PORT, () => console.log(`Backend listening on port ${PORT}`));
