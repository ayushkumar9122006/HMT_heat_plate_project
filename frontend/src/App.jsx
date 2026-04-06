import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// ─── Default BC state ────────────────────────────────────────────
const defaultBC = (T = 0) => ({ type: 'fixed', T, h: 25,  Tinf: 20 });

// ─── Predict from regression coefficients ────────────────────────
function predictFromCoeffs(coeffs, m) {
  if (!coeffs) return { iters: '--', time: '--' };
  const { iter_m, iter_c, time_coeffs, time_c } = coeffs;
  const predIters = Math.round(iter_m * m + iter_c);
  const [, c1 = 0, c2 = 0, c3 = 0] = time_coeffs;
  const predTime = time_c + c1 * m + c2 * m ** 2 + c3 * m ** 3;
  return {
    iters: predIters > 0 ? predIters : '--',
    time:  predTime  > 0 ? predTime.toFixed(4) : '--',
  };
}

// ─── Plotly layout helpers (light mode) ──────────────────────────
const heatmapLayout = () => ({
  width: 660, height: 660,
  paper_bgcolor: '#ffffff',
  plot_bgcolor:  '#ffffff',
  xaxis: {
    constrain: 'domain',
    title: { text: 'x →', font: { color: '#6b7280', size: 11 } },
    tickfont: { color: '#9ca3af' }, gridcolor: '#e5e7eb', zerolinecolor: '#d1d5db',
  },
  yaxis: {
    scaleanchor: 'x', scaleratio: 1,
    title: { text: 'y →', font: { color: '#6b7280', size: 11 } },
    tickfont: { color: '#9ca3af' }, gridcolor: '#e5e7eb', zerolinecolor: '#d1d5db',
  },
  margin: { l: 55, r: 20, t: 16, b: 55 },
});

const scatterLayout = (title, xLabel, yLabel) => ({
  title: { text: title, font: { family: 'Inter, system-ui, sans-serif', size: 12, color: '#03070cff' } },
  width: 390, height: 300,
  paper_bgcolor: '#ffffff',
  plot_bgcolor:  '#f9fafb',
  xaxis: { title: { text: xLabel, font: { color: '#6b7280', size: 10 } }, tickfont: { color: '#9ca3af' }, gridcolor: '#e5e7eb', zerolinecolor: '#d1d5db' },
  yaxis: { title: { text: yLabel, font: { color: '#6b7280', size: 10 } }, tickfont: { color: '#9ca3af' }, gridcolor: '#e5e7eb', zerolinecolor: '#d1d5db' },
  margin: { t: 44, l: 64, r: 16, b: 60 },
  legend: { font: { color: '#6b7280', size: 10 }, bgcolor: 'rgba(0,0,0,0)' },
  showlegend: true,
});

// ─── Isothermal traces (heatmap + contour overlay) ────────────────
const isoTraces = (z, colorscale) => [
  {
    z, type: 'heatmap', colorscale,
    colorbar: {
      tickfont: { color: '#010409ff', size: 10 },
      outlinecolor: '#010307ff',
      bgcolor: '#f9f8f8ff',
      len: 0.85,
    },
    hovertemplate: 'x: %{x}<br>y: %{y}<br>T: %{z}<extra></extra>',
  },
  {
    z, type: 'contour',
    contours: {
      coloring: 'lines',
      showlabels: true,
      labelfont: { size: 10, color: '#000000', family: 'JetBrains Mono, monospace' },
    },
    line: { width: 2.5,
       smoothing: 0.85,
      color:'#000000'  },
    showscale: false,
    opacity: 0.9,
    hoverinfo: 'skip',
  }, 
];

// ══════════════════════════════════════════════════════════════════
//  BC INPUT COMPONENT  — one per boundary side
// ══════════════════════════════════════════════════════════════════
function BCSideInput({ label, value, onChange, accentColor ,plateK}) {
  const isConv = value.type === 'convective';

  return (
    <div className="bc-side-card" style={{ '--accent': accentColor }}>
      <div className="bc-side-head">
        <span className="bc-side-label">{label}</span>
        <div className="bc-type-toggle">
          <button
            className={`bc-toggle-btn ${!isConv ? 'active' : ''}`}
            onClick={() => onChange({ ...value, type: 'fixed' })}
          >
            Fixed T
          </button>
          <button
            className={`bc-toggle-btn ${isConv ? 'active' : ''}`}
            onClick={() => onChange({ ...value, type: 'convective' })}
          >
            Convective
          </button>
        </div>
      </div>

      {!isConv ? (
        <div className="input-group">
          <label>Temperature (K)</label>
          <input
            type="number"
            value={value.T}
            onChange={e => onChange({ ...value, T: Number(e.target.value) })}
          />
        </div>
      ) : (
        <div className="conv-inputs">
          <div className="input-group">
            <label>T∞ — Ambient (K)</label>
            <input
              type="number"
              value={value.T}
              onChange={e => onChange({ ...value, T: Number(e.target.value) })}
            />
          </div>
          <div className="conv-row">
            <div className="input-group">
              <label>h  (W/m²·K)</label>
              <input
                type="number"
                value={value.h}
                min={0.1}
                onChange={e => onChange({ ...value, h: Number(e.target.value) })}
              />
            </div>
            {/* <div className="input-group">
              <label>k  (W/m·K)</label>
              <input
                type="number"
                value={value.k}
                min={0.1}
                onChange={e => onChange({ ...value, k: Number(e.target.value) })}
              />
            </div> */}
          </div>
          <div className="bi-display">
            Bi = h/k = <strong>{(value.h / plateK).toFixed(3)}</strong>
          </div>
        </div>
      )}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════
//  MAIN APP
// ══════════════════════════════════════════════════════════════════
export default function App() {
  const [m, setM]             = useState(50);
  const [plateK, setPlateK] = useState(200);
  const [bcs, setBcs]         = useState({
    top:    defaultBC(100),
    bottom: defaultBC(0),
    left:   defaultBC(0),
    right:  defaultBC(0),
  });
  const [results, setResults]     = useState(null);
  const [loading, setLoading]     = useState(false);
  const [progress, setProgress]   = useState(0);
  const [status, setStatus]       = useState('checking');
  const [regData, setRegData]     = useState(null);
  const [prediction, setPrediction] = useState({ iters: '--', time: '--' });
  const [query, setQuery]         = useState({ x: 0, y: 0 });
  const [queryResult, setQueryResult] = useState(null);

  // Health check
  useEffect(() => {
    (async () => {
      try {
        const res = await axios.get(`${API}/api/health`);
        if (res.data.status === 'online') { setStatus('online'); fetchReg(); }
        else setStatus('offline');
      } catch { setStatus('offline'); }
    })();
  }, []);

  // Live prediction
  useEffect(() => {
    if (regData?.coeffs) setPrediction(predictFromCoeffs(regData.coeffs, m));
  }, [m, regData]);

  const fetchReg = useCallback(async () => {
    try {
      const r = await axios.get(`${API}/api/regression`);
      if (!r.data.error) setRegData(r.data);
    } catch {}
  }, []);

  // ── Build payload — maps frontend BC state to backend BCSpec ──
  const buildPayload = () => ({
    m,
    top:    { type: bcs.top.type,    T: bcs.top.T,    h: bcs.top.h,    k: plateK  },
    bottom: { type: bcs.bottom.type, T: bcs.bottom.T, h: bcs.bottom.h, k: plateK },
    left:   { type: bcs.left.type,   T: bcs.left.T,   h: bcs.left.h,   k: plateK  },
    right:  { type: bcs.right.type,  T: bcs.right.T,  h: bcs.right.h,  k: plateK  },
  });

  // ── Solve ──
  const solve = async () => {
    setLoading(true); setProgress(0); setResults(null); setQueryResult(null);
    const poll = setInterval(async () => {
      try { const r = await axios.get(`${API}/api/progress`); setProgress(r.data.progress); }
      catch {}
    }, 200);
    try {
      const res = await axios.post(`${API}/api/calculate`, buildPayload());
      clearInterval(poll); setProgress(100);
      setResults(res.data);
      fetchReg();
    } catch (e) {
      clearInterval(poll);
      console.error('Solve error:', e);
    }
    setLoading(false);
  };

  // ── Probe ──
  const probe = () => {
    if (!results) return;
    const { x, y } = query;
    if (x < 0 || x > m || y < 0 || y > m) {
      alert(`Out of bounds! Valid range: 0 – ${m}`); return;
    }
    const tFDM = results.fdm[y][x];
    const tANA = results.analytic[y][x];
    const ref  = Math.abs(tANA) > 0.01 ? Math.abs(tANA) : 1.0;
    const acc  = Math.max(0, 100 * (1 - Math.abs(tFDM - tANA) / ref));
    setQueryResult({
      f:    tFDM.toFixed(4),
      a:    tANA.toFixed(4),
      acc:  acc.toFixed(2),
      diff: Math.abs(tFDM - tANA).toFixed(4),
    });
  };

  const hasConvective = Object.values(bcs).some(b => b.type === 'convective');

  // ── RENDER ──────────────────────────────────────────────────────
  return (
    <div className="dashboard">

      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <span className="header-icon">🌡</span>
          <div>
            <h1>Thermal Analysis Dashboard</h1>
            <p className="header-sub">2D Steady-State · SOR/FDM vs Fourier Series · Fixed &amp; Convective BCs</p>
          </div>
        </div>
        <div className={`status-badge ${status}`}>
          <span className="status-dot" />
          {status === 'online' ? 'System Online' : status === 'offline' ? 'System Offline' : 'Connecting…'}
        </div>
      </header>

      {/* ── Top row ── */}
      <div className="top-row">

        {/* ── Config panel ── */}
        <div className="card config-panel">
          <h3 className="card-title"><span className="card-title-icon">⚙</span> Parameters</h3>

          <div className="input-group" style={{ marginBottom: 18 }}>
            <label>Grid Size (m)</label>
            <input type="number" value={m} min={10} max={200}
              onChange={e => setM(Number(e.target.value))} />
          </div>

          <div className="input-group">
      <label>Plate K (W/m·K)</label>
      <input type="number" value={plateK} onChange={e => setPlateK(Number(e.target.value))} />
    </div>

          <p className="bc-section-label">Boundary Conditions</p>

          {/* BC diagram grid */}
          <div className="bc-layout">
            {/* Top */}
            <div className="bc-top">
              <BCSideInput
                label="TOP"
                value={bcs.top}
                onChange={v => setBcs({ ...bcs, top: v })}
                accentColor="#f87171"
                plateK={plateK}
              />
            </div>
            {/* Middle row: Left + plate preview + Right */}
            <div className="bc-middle">
              <BCSideInput
                label="LEFT"
                value={bcs.left}
                onChange={v => setBcs({ ...bcs, left: v })}
                accentColor="#fbbf24"
                plateK={plateK}
              />
              </div>
              
              <div className="bc-middle">
              <BCSideInput
                label="RIGHT"
                value={bcs.right}
                onChange={v => setBcs({ ...bcs, right: v })}
                accentColor="#34d399"
                plateK={plateK}
              />
            </div>
            {/* Bottom */}
            <div className="bc-bottom">
              <BCSideInput
                label="BOTTOM"
                value={bcs.bottom}
                onChange={v => setBcs({ ...bcs, bottom: v })}
                accentColor="#22d3ee"
                plateK={plateK}
              />
            </div>
          </div>

          {hasConvective && (
            <div className="conv-notice">
              ⚡ Convective BC detected — analytical solution uses transcendental eigenvalues (Fourier–Robin series).
            </div>
          )}

          <button
            className="execute-btn"
            onClick={solve}
            disabled={loading || status !== 'online'}
          >
            {loading ? `Solving… ${progress}%` : '▶  Execute Analysis'}
          </button>

          {loading && (
            <div className="progress-wrap">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
              <span className="progress-pct">{progress}%</span>
            </div>
          )}
        </div>

        {/* ── Metrics ── */}
        <div className="metrics-grid">
          {[
            { label: 'Est. Iterations',  value: prediction.iters, icon: '∿', color: 'violet' },
            { label: 'Actual Iterations',value: results?.iters ?? '--', icon: '#', color: 'amber'  },
            { label: 'Est. Solve Time',  value: prediction.time !== '--' ? `${prediction.time}s` : '--', icon: '⏱', color: 'cyan' },
            { label: 'Actual Solve Time',value: results ? `${results.time.toFixed(4)}s` : '--', icon: '⚡', color: 'green' },
            { label: 'RMSE',             value: results ? results.rmse.toFixed(4) : '--', icon: 'Δ', color: 'red'   },
            { label: 'Validation Score', value: results ? `${results.v_score.toFixed(1)}%` : '--', icon: '✓', color: 'lime'  },
          ].map(({ label, value, icon, color }) => (
            <div key={label} className={`metric-card metric-${color}`}>
              <span className="metric-icon">{icon}</span>
              <span className="metric-value">{value}</span>
              <span className="metric-label">{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Results ── */}
      {results && (
        <div className="results-area">

          {/* Heatmaps */}
          <div className="card plot-row">
            <div className="heatmap-wrap">
              <p className="plot-label fdm-label">
                Numerical — SOR / FDM
                <span className="plot-sublabel">
                  {results.iters} iterations · {results.time.toFixed(3)}s
                </span>
              </p>
              <Plot
                data={isoTraces(results.fdm, 'Jet')}
                layout={heatmapLayout()}
                config={{ displayModeBar: false }}
              />
            </div>
            <div className="plot-divider" />
            <div className="heatmap-wrap">
              <p className="plot-label ana-label">
                Analytical — Fourier Series
                <span className="plot-sublabel">
                  {results.has_convective ? 'Robin eigenvalues' : 'Classic Fourier'}
                </span>
              </p>
              <Plot
                data={isoTraces(results.analytic, 'Jet')}
                layout={heatmapLayout()}
                config={{ displayModeBar: false }}
              />
            </div>
          </div>

          {/* Bottom row */}
          <div className="bottom-row">

            {/* Probe */}
            <div className="card probe-card">
              <h3 className="card-title"><span className="card-title-icon">⊕</span> Coordinate Probe</h3>
              <p className="probe-hint">Click any grid point to compare FDM vs Analytical temperature.</p>

              <div className="probe-inputs">
                <div className="input-group">
                  <label>X — Column (0 – {m})</label>
                  <input type="number" min={0} max={m} placeholder="0"
                    value={query.x}
                    onChange={e => setQuery({ ...query, x: Number(e.target.value) })} />
                </div>
                <div className="input-group">
                  <label>Y — Row (0 – {m})</label>
                  <input type="number" min={0} max={m} placeholder="0"
                    value={query.y}
                    onChange={e => setQuery({ ...query, y: Number(e.target.value) })} />
                </div>
                <button className="probe-btn" onClick={probe}>Run Query</button>
              </div>

              {queryResult && (
                <div className="probe-results">
                  <div className="probe-result-row fdm-result">
                    <span>FDM (SOR)</span>
                    <strong>{queryResult.f} K</strong>
                  </div>
                  <div className="probe-result-row ana-result">
                    <span>Analytical (Fourier)</span>
                    <strong>{queryResult.a} K</strong>
                  </div>
                  {/* <div className="probe-result-row diff-result">
                    <span>Absolute Difference</span>
                    <strong>{queryResult.diff} K</strong>
                  </div>
                  <div className="probe-accuracy-bar">
                    <div className="accuracy-fill" style={{ width: `${Math.min(100, queryResult.acc)}%` }} />
                    <span className="accuracy-label">FDM Accuracy: {queryResult.acc}%</span>
                  </div> */}
                </div>
              )}
            </div>

            {/* Regression charts */}
            <div className="card regression-card">
              <h3 className="card-title"><span className="card-title-icon">📈</span> Historical Performance</h3>
              {regData ? (
                <div className="reg-plots">
                  <Plot
                    data={[
                      { x: regData.m_values, y: regData.iter_values, mode: 'markers', name: 'Past Runs', marker: { color: '#818cf8', size: 7, opacity: 0.7 } },
                      { x: regData.m_curve,  y: regData.iter_curve,  mode: 'lines',   name: 'Linear fit', line: { color: '#c084fc', width: 2, dash: 'dash' } },
                      ...(results ? [{ x: [m], y: [results.iters], mode: 'markers', name: 'This run', marker: { color: '#f87171', size: 12, symbol: 'diamond' } }] : []),
                    ]}
                    layout={scatterLayout('Iterations vs Grid Size', 'Grid Size (m)', 'Iterations')}
                    config={{ displayModeBar: false }}
                  />
                  <Plot
                    data={[
                      { x: regData.m_values, y: regData.time_values, mode: 'markers', name: 'Past Runs', marker: { color: '#34d399', size: 7, opacity: 0.7 } },
                      { x: regData.m_curve,  y: regData.time_curve,  mode: 'lines',   name: 'Cubic fit',  line: { color: '#6ee7b7', width: 2, dash: 'dash' } },
                      ...(results ? [{ x: [m], y: [results.time], mode: 'markers', name: 'This run', marker: { color: '#fbbf24', size: 12, symbol: 'diamond' } }] : []),
                    ]}
                    layout={scatterLayout('Runtime vs Grid Size', 'Grid Size (m)', 'Time (s)')}
                    config={{ displayModeBar: false }}
                  />
                </div>
              ) : (
                <div className="no-data">No historical data yet. Run a solve to begin.</div>
              )}
            </div>

          </div>
        </div>
      )}
    </div>
  );
}
