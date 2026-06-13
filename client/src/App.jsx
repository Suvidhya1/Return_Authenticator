import { useState } from "react";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:5000";

const CATEGORIES = [
  "electronics", "clothing", "home", "beauty", "toys", "books", "sports",
];
const REASONS = [
  "defective", "not_as_described", "wrong_item", "changed_mind",
  "better_price_found", "no_longer_needed", "damaged_in_transit",
];

const ACTION_LABELS = {
  instant_refund: { text: "Instant Refund Approved", tone: "good" },
  wait_for_item: { text: "Refund After Item Received", tone: "warn" },
  route_to_agent: { text: "Flagged — Routed to Agent", tone: "bad" },
};

export default function App() {
  const [form, setForm] = useState({
    orderValue: 120,
    productCategory: "electronics",
    customerReturnRate: 0.2,
    daysSincePurchase: 7,
    priorOrders: 12,
    returnReason: "defective",
    isHighValue: 0,
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const update = (key, value) => setForm((f) => ({ ...f, [key]: value }));

  const submit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API}/api/returns`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) throw new Error((await res.json()).error || "Request failed");
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header>
        <h1>ReturnAuthenticator</h1>
        <p className="sub">Return &amp; Refund Predictor</p>
      </header>

      <div className="grid">
        <section className="card">
          <h2>Return Request</h2>

          <label>
            Order Value ($)
            <input
              type="number"
              value={form.orderValue}
              onChange={(e) => update("orderValue", Number(e.target.value))}
            />
          </label>

          <label>
            Product Category
            <select
              value={form.productCategory}
              onChange={(e) => update("productCategory", e.target.value)}
            >
              {CATEGORIES.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </label>

          <label>
            Customer Return Rate (0–1)
            <input
              type="number" step="0.01" min="0" max="1"
              value={form.customerReturnRate}
              onChange={(e) => update("customerReturnRate", Number(e.target.value))}
            />
          </label>

          <label>
            Days Since Purchase
            <input
              type="number"
              value={form.daysSincePurchase}
              onChange={(e) => update("daysSincePurchase", Number(e.target.value))}
            />
          </label>

          <label>
            Prior Orders
            <input
              type="number"
              value={form.priorOrders}
              onChange={(e) => update("priorOrders", Number(e.target.value))}
            />
          </label>

          <label>
            Return Reason
            <select
              value={form.returnReason}
              onChange={(e) => update("returnReason", e.target.value)}
            >
              {REASONS.map((r) => (
                <option key={r} value={r}>{r.replace(/_/g, " ")}</option>
              ))}
            </select>
          </label>

          <label className="checkbox">
            <input
              type="checkbox"
              checked={form.isHighValue === 1}
              onChange={(e) => update("isHighValue", e.target.checked ? 1 : 0)}
            />
            High-value item
          </label>

          <button onClick={submit} disabled={loading}>
            {loading ? "Scoring…" : "Score Return"}
          </button>
          {error && <p className="error">{error}</p>}
        </section>

        <section className="card">
          <h2>Prediction</h2>
          {!result && <p className="muted">Submit a request to see the result.</p>}
          {result && (
            <div className="result">
              <div className={`action ${ACTION_LABELS[result.recommended_action].tone}`}>
                {ACTION_LABELS[result.recommended_action].text}
              </div>

              <p className="label-line">
                Classification: <strong>{result.label}</strong>
              </p>

              <div className="score">
                <span>Fraud score</span>
                <div className="bar">
                  <div
                    className="fill"
                    style={{ width: `${result.fraud_score * 100}%` }}
                  />
                </div>
                <span>{(result.fraud_score * 100).toFixed(1)}%</span>
              </div>

              <h3>Class probabilities</h3>
              <ul className="probs">
                {Object.entries(result.probabilities).map(([k, v]) => (
                  <li key={k}>
                    <span>{k}</span>
                    <span>{(v * 100).toFixed(1)}%</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
