import React, { useState, useEffect, useRef } from 'react';
import { Toaster, toast } from 'react-hot-toast';
import { Shield, Zap, ChevronDown, Activity, AlertTriangle, Cpu, Layers, BarChart3, Table2, Fingerprint } from 'lucide-react';

import FileUpload from './FileUpload';
import RiskScoreCard from './RiskScoreCard';
import ClusterScatterPlot from './ClusterScatterPlot';
import LabelHeatmap from './LabelHeatmap';
import InfluenceScoreChart from './InfluenceScoreChart';
import FlaggedSamplesTable from './FlaggedSamplesTable';
import LayerBreakdown from './LayerBreakdown';
import DomainProfileCard from './DomainProfileCard';
import RiskReportDownload from './RiskReportDownload';

const API_BASE = 'http://127.0.0.1:8000';

function Dashboard() {
  const [files, setFiles] = useState({ csv: null, model: null });
  const [domain, setDomain] = useState('general');
  const [labelColumn, setLabelColumn] = useState('');
  const [profiles, setProfiles] = useState([]);
  const [scanning, setScanning] = useState(false);
  const [results, setResults] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [health, setHealth] = useState(null);
  const [scanProgress, setScanProgress] = useState(0);
  const progressRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/profiles`)
      .then(r => r.json())
      .then(data => setProfiles(data.profiles || []))
      .catch(() => {});

    fetch(`${API_BASE}/api/health`)
      .then(r => r.json())
      .then(data => setHealth(data))
      .catch(() => setHealth({ status: 'offline' }));
  }, []);

  // Simulated progress during scan
  useEffect(() => {
    if (scanning) {
      setScanProgress(0);
      const steps = [
        { target: 15, delay: 500 },
        { target: 35, delay: 2000 },
        { target: 55, delay: 5000 },
        { target: 75, delay: 10000 },
        { target: 88, delay: 20000 },
      ];
      const timers = steps.map(s =>
        setTimeout(() => setScanProgress(s.target), s.delay)
      );
      return () => timers.forEach(clearTimeout);
    } else {
      setScanProgress(results ? 100 : 0);
    }
  }, [scanning, results]);

  const handleScan = async () => {
    if (!files.csv) {
      toast.error('Please upload a CSV dataset first.');
      return;
    }

    setScanning(true);
    setResults(null);

    const formData = new FormData();
    formData.append('dataset', files.csv);
    if (files.model) {
      formData.append('model_file', files.model);
    }
    formData.append('domain', domain);
    if (labelColumn.trim()) {
      formData.append('label_column', labelColumn.trim());
    }

    try {
      toast.loading('Running multi-layer detection engine...', { id: 'scan' });

      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
      setActiveTab('overview');

      if (data.risk_level === 'LOW') {
        toast.success('Analysis complete — dataset appears clean!', { id: 'scan' });
      } else if (data.risk_level === 'CRITICAL' || data.risk_level === 'HIGH') {
        toast.error(`⚠ ${data.n_flagged} suspicious samples detected!`, { id: 'scan' });
      } else {
        toast('Analysis complete — some anomalies found.', { id: 'scan', icon: '⚠️' });
      }
    } catch (error) {
      console.error('Scan failed:', error);
      toast.error(`Scan failed: ${error.message}`, { id: 'scan' });
    } finally {
      setScanning(false);
    }
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: <Layers size={14} /> },
    { id: 'clusters', label: 'Clusters', icon: <Fingerprint size={14} /> },
    { id: 'heatmap', label: 'Heatmap', icon: <BarChart3 size={14} /> },
    { id: 'influence', label: 'Influence', icon: <Activity size={14} /> },
    { id: 'samples', label: 'Flagged', icon: <Table2 size={14} /> },
  ];

  return (
    <div className="min-h-screen" style={{ background: 'var(--gradient-bg)', position: 'relative', zIndex: 1 }}>
      {/* Noise overlay */}
      <div className="noise-overlay" />

      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: 'rgba(12, 18, 34, 0.95)',
            backdropFilter: 'blur(12px)',
            color: 'var(--text-primary)',
            border: '1px solid var(--border)',
            fontSize: '13px',
            borderRadius: 'var(--radius-md)',
            boxShadow: 'var(--shadow-md)',
          },
        }}
      />

      {/* ─── Header ─── */}
      <header style={{
        borderBottom: '1px solid var(--border)',
        background: 'rgba(6, 10, 19, 0.85)',
        backdropFilter: 'blur(16px) saturate(180%)',
        position: 'sticky',
        top: 0,
        zIndex: 50,
        padding: '0 24px',
      }}>
        <div style={{
          maxWidth: '1440px',
          margin: '0 auto',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          height: '64px',
        }}>
          <div className="flex items-center gap-3">
            <div style={{
              background: 'var(--gradient-primary)',
              padding: '10px',
              borderRadius: 'var(--radius-md)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 20px rgba(99, 102, 241, 0.25)',
            }}>
              <Shield size={18} color="white" />
            </div>
            <div>
              <h1 style={{
                fontSize: '17px',
                fontWeight: 800,
                color: 'var(--text-bright)',
                letterSpacing: '-0.02em',
                lineHeight: 1.2,
              }}>
                AI PoisonGuard
              </h1>
              <p style={{
                fontSize: '11px',
                color: 'var(--text-muted)',
                letterSpacing: '0.02em',
              }}>
                Adversarial Training Data Poisoning Detector
              </p>
            </div>
          </div>

          {/* Status badges */}
          <div className="flex items-center gap-3">
            {health && (
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <div className="status-dot" style={{
                    background: health.status === 'healthy' ? 'var(--accent-green)' : 'var(--accent-red)',
                    color: health.status === 'healthy' ? 'var(--accent-green)' : 'var(--accent-red)',
                  }} />
                  <span style={{ color: 'var(--text-muted)', fontSize: '12px', fontWeight: 500 }}>
                    {health.status === 'healthy' ? 'Engine Online' : 'Engine Offline'}
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  {health.umap_available && (
                    <span className="tag tag-low" style={{ fontSize: '9px', padding: '2px 8px' }}>UMAP</span>
                  )}
                  {health.hdbscan_available && (
                    <span className="tag tag-low" style={{ fontSize: '9px', padding: '2px 8px' }}>HDBSCAN</span>
                  )}
                  {health.art_available && (
                    <span className="tag tag-low" style={{ fontSize: '9px', padding: '2px 8px' }}>ART</span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      <main style={{ maxWidth: '1440px', margin: '0 auto', padding: '28px 24px' }}>
        {/* ─── Controls Row ─── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Upload Card */}
          <div className="lg:col-span-2 glass-card p-6 animate-fadeIn">
            <div className="flex items-center gap-2 mb-5">
              <Cpu size={15} style={{ color: 'var(--accent-indigo)' }} />
              <h2 style={{
                fontSize: '12px',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                color: 'var(--text-muted)',
              }}>
                Model & Dataset Ingestion
              </h2>
            </div>
            <FileUpload onFilesSelected={setFiles} disabled={scanning} />

            <div className="flex items-end gap-4 mt-5">
              {/* Domain selector */}
              <div className="relative flex-1">
                <label style={{
                  fontSize: '11px',
                  fontWeight: 600,
                  color: 'var(--text-muted)',
                  display: 'block',
                  marginBottom: '6px',
                  letterSpacing: '0.04em',
                  textTransform: 'uppercase',
                }}>
                  Domain Profile
                </label>
                <div className="relative">
                  <select
                    id="domain-select"
                    value={domain}
                    onChange={(e) => setDomain(e.target.value)}
                    disabled={scanning}
                    style={{
                      width: '100%',
                      padding: '10px 36px 10px 14px',
                      borderRadius: 'var(--radius-md)',
                      background: 'var(--bg-secondary)',
                      border: '1px solid var(--border)',
                      color: 'var(--text-primary)',
                      fontSize: '13px',
                      fontWeight: 500,
                      cursor: scanning ? 'not-allowed' : 'pointer',
                      outline: 'none',
                      appearance: 'none',
                      fontFamily: 'Inter, sans-serif',
                      transition: 'border-color 0.2s ease',
                    }}
                  >
                    {profiles.map(p => (
                      <option key={p.id} value={p.id}>
                        {p.icon} {p.name}
                      </option>
                    ))}
                    {profiles.length === 0 && (
                      <>
                        <option value="general">🔍 General Purpose</option>
                        <option value="upi_fraud">₹ UPI Fraud Detection</option>
                        <option value="credit_scoring">📊 Credit Scoring</option>
                        <option value="kyc_govt_welfare">🏛️ KYC / Govt Welfare</option>
                      </>
                    )}
                  </select>
                  <ChevronDown size={14}
                    className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none"
                    style={{ color: 'var(--text-muted)' }}
                  />
                </div>
              </div>

              {/* Label column */}
              <div style={{ minWidth: '130px' }}>
                <label style={{
                  fontSize: '11px',
                  fontWeight: 600,
                  color: 'var(--text-muted)',
                  display: 'block',
                  marginBottom: '6px',
                  letterSpacing: '0.04em',
                  textTransform: 'uppercase',
                }}>
                  Label Column
                </label>
                <input
                  id="label-column-input"
                  type="text"
                  value={labelColumn}
                  onChange={(e) => setLabelColumn(e.target.value)}
                  placeholder="auto-detect"
                  disabled={scanning}
                  style={{
                    width: '100%',
                    padding: '10px 14px',
                    borderRadius: 'var(--radius-md)',
                    background: 'var(--bg-secondary)',
                    border: '1px solid var(--border)',
                    color: 'var(--text-primary)',
                    fontSize: '13px',
                    outline: 'none',
                    fontFamily: 'Inter, sans-serif',
                    transition: 'border-color 0.2s ease',
                  }}
                />
              </div>

              {/* Scan button */}
              <button
                id="scan-button"
                onClick={handleScan}
                disabled={!files.csv || scanning}
                className="btn-primary flex items-center justify-center gap-2"
                style={{ padding: '10px 28px', minWidth: '180px', whiteSpace: 'nowrap' }}
              >
                {scanning ? (
                  <>
                    <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
                    <span>Analyzing…</span>
                  </>
                ) : (
                  <>
                    <Zap size={16} />
                    <span>Run Detection</span>
                  </>
                )}
              </button>
            </div>

            {/* Scan progress bar */}
            {scanning && (
              <div style={{ marginTop: '16px' }}>
                <div className="scan-progress">
                  <div className="scan-progress-bar" style={{ width: `${scanProgress}%`, transition: 'width 1s ease' }} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px' }}>
                  {['Statistical', 'Spectral', 'Clustering', 'ART', 'Influence', 'Backdoor'].map((layer, i) => (
                    <span key={layer} style={{
                      fontSize: '10px',
                      fontWeight: 600,
                      color: scanProgress > (i + 1) * 14 ? 'var(--accent-green)' : 'var(--text-muted)',
                      transition: 'color 0.3s ease',
                      letterSpacing: '0.02em',
                    }}>
                      {scanProgress > (i + 1) * 14 ? '✓ ' : ''}{layer}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Risk Score or Placeholder */}
          {results ? (
            <RiskScoreCard
              riskScore={results.risk_score}
              riskLevel={results.risk_level}
              nFlagged={results.n_flagged}
              nWarnings={results.n_warnings || 0}
              nSamples={results.n_samples}
              status={results.status}
              ensembleThreshold={results.ensemble_threshold || 2}
            />
          ) : (
            <div className="glass-card p-6 flex flex-col items-center justify-center text-center animate-fadeIn stagger-1">
              <div style={{
                padding: '20px',
                borderRadius: '50%',
                background: 'rgba(99, 102, 241, 0.06)',
                border: '1px solid rgba(99, 102, 241, 0.12)',
                marginBottom: '16px',
              }}>
                <Shield size={40} style={{ color: 'var(--accent-indigo)', opacity: 0.4 }} />
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-muted)', fontWeight: 500 }}>
                Upload a dataset and run detection
              </p>
              <p style={{ fontSize: '11px', color: 'var(--text-muted)', opacity: 0.5, marginTop: '4px' }}>
                Supports .csv datasets and .pkl / .h5 / .onnx models
              </p>
            </div>
          )}
        </div>

        {/* ─── Summary Bar ─── */}
        {results && (
          <div className="glass-card p-5 mb-6 animate-slideUp flex items-center justify-between gap-4"
            style={{
              borderLeft: `3px solid ${
                results.risk_level === 'LOW' ? 'var(--accent-green)'
                : results.risk_level === 'MEDIUM' ? 'var(--accent-yellow)'
                : results.risk_level === 'HIGH' ? 'var(--accent-orange)'
                : 'var(--accent-red)'
              }`
            }}>
            <p style={{
              fontSize: '13px',
              flex: 1,
              color: 'var(--text-secondary)',
              lineHeight: 1.7,
            }}>
              {results.summary}
            </p>
            <RiskReportDownload results={results} />
          </div>
        )}

        {/* ─── Results Tabs ─── */}
        {results && (
          <>
            <div style={{
              display: 'inline-flex',
              gap: '2px',
              padding: '4px',
              borderRadius: 'var(--radius-lg)',
              background: 'rgba(12, 18, 34, 0.6)',
              border: '1px solid var(--border)',
              marginBottom: '24px',
            }}>
              {tabs.map(tab => (
                <button
                  key={tab.id}
                  id={`tab-${tab.id}`}
                  onClick={() => setActiveTab(tab.id)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '8px 16px',
                    borderRadius: 'var(--radius-md)',
                    fontSize: '12px',
                    fontWeight: 600,
                    background: activeTab === tab.id ? 'var(--gradient-primary)' : 'transparent',
                    color: activeTab === tab.id ? 'white' : 'var(--text-muted)',
                    cursor: 'pointer',
                    border: 'none',
                    transition: 'all 0.2s ease',
                    letterSpacing: '0.01em',
                    boxShadow: activeTab === tab.id ? '0 4px 15px rgba(99, 102, 241, 0.25)' : 'none',
                  }}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <div className="animate-fadeIn" key={activeTab}>
              {activeTab === 'overview' && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <div className="lg:col-span-2">
                    <LayerBreakdown
                      statistical={results.statistical}
                      spectral={results.spectral}
                      clustering={results.clustering}
                      art={results.art}
                      influence={results.influence}
                      backdoor={results.backdoor}
                    />
                  </div>
                  <DomainProfileCard
                    profileInfo={results.profile_info}
                    modelInfo={results.model_info}
                  />
                </div>
              )}

              {activeTab === 'clusters' && (
                <ClusterScatterPlot scatterData={results.scatter_data} />
              )}

              {activeTab === 'heatmap' && (
                <LabelHeatmap heatmapData={results.label_heatmap_data} />
              )}

              {activeTab === 'influence' && (
                <InfluenceScoreChart influenceScores={results.influence_scores} />
              )}

              {activeTab === 'samples' && (
                <FlaggedSamplesTable flaggedSamples={results.flagged_samples} />
              )}
            </div>
          </>
        )}

        {/* ─── Empty State ─── */}
        {!results && !scanning && (
          <div style={{ textAlign: 'center', padding: '80px 20px' }} className="animate-fadeIn">
            <div style={{
              display: 'inline-flex',
              padding: '24px',
              borderRadius: '50%',
              background: 'rgba(99, 102, 241, 0.06)',
              border: '1px solid rgba(99, 102, 241, 0.12)',
              marginBottom: '28px',
              position: 'relative',
            }}>
              <div style={{
                position: 'absolute',
                inset: '-6px',
                borderRadius: '50%',
                border: '1px solid rgba(99, 102, 241, 0.08)',
                animation: 'ripple 3s ease-out infinite',
              }} />
              <Shield size={44} style={{ color: 'var(--accent-indigo)', opacity: 0.5 }} />
            </div>
            <h2 style={{
              fontSize: '26px',
              fontWeight: 800,
              marginBottom: '12px',
              color: 'var(--text-bright)',
              letterSpacing: '-0.02em',
            }}>
              AI PoisonGuard Detection Engine
            </h2>
            <p style={{
              fontSize: '14px',
              maxWidth: '520px',
              margin: '0 auto 36px',
              color: 'var(--text-muted)',
              lineHeight: 1.7,
            }}>
              Multi-layer adversarial training data poisoning detector for Indian fintech
              and government ML systems. Upload a CSV dataset to begin analysis.
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              {[
                { label: 'Z-score + IQR', desc: 'Statistical Analysis', color: '#38bdf8', icon: '📊' },
                { label: 'SVD Spectral', desc: 'Signature Detection', color: '#818cf8', icon: '🔬' },
                { label: 'PyTorch + UMAP', desc: 'Activation Clustering', color: '#22d3ee', icon: '🧠' },
                { label: 'IBM ART', desc: 'Validation Layer', color: '#f97316', icon: '🛡️' },
                { label: 'TracIn / LOO', desc: 'Influence Functions', color: '#f472b6', icon: '📈' },
                { label: 'Neural Cleanse', desc: 'Backdoor Scanning', color: '#a78bfa', icon: '🔓' },
              ].map(({ label, desc, color, icon }, i) => (
                <div key={label}
                  className={`glass-card animate-fadeIn stagger-${i + 1}`}
                  style={{
                    padding: '16px 24px',
                    textAlign: 'center',
                    minWidth: '150px',
                    borderTop: `2px solid ${color}`,
                  }}>
                  <div style={{ fontSize: '20px', marginBottom: '6px' }}>{icon}</div>
                  <p style={{ fontWeight: 700, color, fontSize: '13px', letterSpacing: '-0.01em' }}>{label}</p>
                  <p style={{ color: 'var(--text-muted)', marginTop: '2px', fontSize: '11px' }}>{desc}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Scanning animation */}
        {scanning && !results && (
          <div style={{ textAlign: 'center', padding: '60px 20px' }} className="animate-fadeIn">
            <div style={{
              position: 'relative',
              display: 'inline-block',
              marginBottom: '28px',
            }}>
              <div className="spinner" style={{
                width: 56,
                height: 56,
                borderWidth: 4,
                margin: '0 auto',
              }} />
              <div style={{
                position: 'absolute',
                inset: '8px',
                borderRadius: '50%',
                border: '2px solid rgba(56, 189, 248, 0.1)',
                borderTopColor: 'var(--accent-purple)',
                animation: 'spin 1.2s linear infinite reverse',
              }} />
            </div>
            <h3 style={{
              fontSize: '18px',
              fontWeight: 700,
              marginBottom: '8px',
              color: 'var(--text-bright)',
            }}>
              Running Multi-Layer Detection
            </h3>
            <p style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
              Statistical → Spectral → Clustering → ART → Influence → Backdoor
            </p>
          </div>
        )}
      </main>

      {/* ─── Footer ─── */}
      <footer style={{
        borderTop: '1px solid var(--border)',
        padding: '20px 24px',
        marginTop: '48px',
        textAlign: 'center',
      }}>
        <p style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: 500 }}>
          AI PoisonGuard v1.0 &nbsp;•&nbsp; Educational Prototype &nbsp;•&nbsp; NMIMS INNOVATHON 2026
        </p>
        <p style={{ fontSize: '10px', color: 'var(--text-muted)', opacity: 0.5, marginTop: '4px' }}>
          CERT-In Cyber Security Framework &nbsp;•&nbsp; RBI ML Governance Guidelines &nbsp;•&nbsp; DPDP Act 2023
        </p>
      </footer>
    </div>
  );
}

export default Dashboard;
