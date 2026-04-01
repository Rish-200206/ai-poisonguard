import React from 'react';
import { FileDown, Shield, AlertTriangle, CheckCircle } from 'lucide-react';

function RiskReportDownload({ results }) {
  if (!results) return null;

  const generateReportHTML = () => {
    const {
      risk_score, risk_level, status, n_samples, n_flagged, flagged_ratio,
      summary, statistical, spectral, clustering, art, influence, backdoor,
      flagged_samples, profile_info, model_info, analysis_time, domain_profile,
    } = results;

    const riskColor = risk_level === 'LOW' ? '#22c55e'
      : risk_level === 'MEDIUM' ? '#eab308'
      : risk_level === 'HIGH' ? '#f97316' : '#ef4444';

    const layerRows = [
      { name: 'Layer 1 — Statistical (Z-score + IQR)', data: statistical },
      { name: 'Layer 2 — Spectral (SVD)', data: spectral },
      { name: 'Layer 3 — Activation Clustering', data: clustering },
      { name: 'Layer 4 — IBM ART', data: art },
      { name: 'Layer 5 — Influence Function', data: influence },
      { name: 'Layer 6 — Backdoor Trigger', data: backdoor },
    ];

    const topFlagged = (flagged_samples || []).slice(0, 30);

    const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>AI PoisonGuard — Poison Risk Report</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; padding: 32px; line-height: 1.6; }
    .container { max-width: 900px; margin: 0 auto; }
    .header { text-align: center; margin-bottom: 40px; padding-bottom: 24px; border-bottom: 2px solid rgba(99,115,155,0.2); }
    .header h1 { font-size: 28px; font-weight: 800; margin-bottom: 8px; }
    .header p { color: #94a3b8; font-size: 14px; }
    .badge { display: inline-block; padding: 6px 16px; border-radius: 100px; font-weight: 700; font-size: 14px; background: ${riskColor}22; color: ${riskColor}; border: 1px solid ${riskColor}44; }
    .score-block { text-align: center; margin: 24px 0 32px; }
    .score-block .score { font-size: 72px; font-weight: 900; color: ${riskColor}; }
    .score-block .label { font-size: 14px; color: #64748b; }
    .section { background: rgba(26, 34, 53, 0.6); border: 1px solid rgba(99,115,155,0.2); border-radius: 12px; padding: 24px; margin-bottom: 20px; }
    .section h2 { font-size: 16px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 16px; color: #94a3b8; }
    .summary { font-size: 15px; color: #cbd5e1; border-left: 3px solid ${riskColor}; padding-left: 16px; margin-bottom: 24px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { text-align: left; padding: 10px 14px; background: #1e293b; color: #94a3b8; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; border-bottom: 1px solid rgba(99,115,155,0.2); }
    td { padding: 10px 14px; border-bottom: 1px solid rgba(99,115,155,0.15); color: #e2e8f0; }
    tr:hover { background: rgba(59,130,246,0.04); }
    .stat { display: inline-block; margin-right: 24px; margin-bottom: 8px; }
    .stat .val { font-size: 20px; font-weight: 700; }
    .stat .lbl { font-size: 12px; color: #64748b; }
    .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid rgba(99,115,155,0.2); color: #475569; font-size: 12px; }
    .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-right: 4px; }
    .tag-stat { background: rgba(59,130,246,0.15); color: #60a5fa; }
    .tag-spec { background: rgba(139,92,246,0.15); color: #a78bfa; }
    .tag-clust { background: rgba(6,182,212,0.15); color: #22d3ee; }
    .tag-art { background: rgba(249,115,22,0.15); color: #fb923c; }
    @media print { body { background: white; color: #1e293b; } .section { border-color: #e2e8f0; } th { background: #f1f5f9; } }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🛡️ AI PoisonGuard — Poison Risk Report</h1>
      <p>Generated ${new Date().toLocaleString()} • Domain: ${profile_info?.name || domain_profile} • ${n_samples} samples analyzed</p>
    </div>

    <div class="score-block">
      <div class="score">${risk_score}</div>
      <div class="label">Risk Score / 100</div>
      <div style="margin-top: 12px;"><span class="badge">${risk_level} RISK</span></div>
    </div>

    <div class="summary">${summary}</div>

    <div class="section">
      <h2>Analysis Overview</h2>
      <div class="stat"><div class="val">${n_samples}</div><div class="lbl">Total Samples</div></div>
      <div class="stat"><div class="val" style="color: #ef4444;">${n_flagged}</div><div class="lbl">Flagged Suspicious</div></div>
      <div class="stat"><div class="val">${(flagged_ratio * 100).toFixed(1)}%</div><div class="lbl">Contamination Rate</div></div>
      <div class="stat"><div class="val">${analysis_time || '—'}s</div><div class="lbl">Analysis Time</div></div>
      ${model_info && !model_info.error ? `<div class="stat"><div class="val">${model_info.type}</div><div class="lbl">Model: ${model_info.filename}</div></div>` : ''}
    </div>

    <div class="section">
      <h2>Detection Layer Results</h2>
      <table>
        <thead><tr><th>Layer</th><th>Flagged</th><th>Rate</th><th>Details</th></tr></thead>
        <tbody>
          ${layerRows.map(l => `<tr>
            <td><strong>${l.name}</strong></td>
            <td>${l.data?.n_flagged ?? '—'}</td>
            <td>${l.data?.flagged_ratio != null ? (l.data.flagged_ratio * 100).toFixed(1) + '%' : '—'}</td>
            <td>${l.data?.layer_name || ''}</td>
          </tr>`).join('')}
        </tbody>
      </table>
    </div>

    ${topFlagged.length > 0 ? `
    <div class="section">
      <h2>Flagged Samples — Ensemble Voting (${Math.min(topFlagged.length, 30)} of ${(flagged_samples || []).length})</h2>
      <table>
        <thead><tr><th>Index</th><th>Risk Score</th><th>Risk Level</th><th>Votes</th><th>Detection Sources</th><th>Label</th></tr></thead>
        <tbody>
          ${topFlagged.map(s => {
            const riskPct = s.ensemble_risk_score ?? (s.score * 100).toFixed(0);
            const riskCat = s.risk_category || (s.n_layers >= 2 ? 'Compromised' : 'Warning');
            const confirmed = s.is_confirmed_poison !== false;
            const catColor = riskCat === 'Critical' ? '#f87171' : riskCat === 'High Risk' ? '#fb923c' : riskCat === 'Compromised' ? '#fbbf24' : '#60a5fa';
            const layerCls = l => l === 'statistical' ? 'tag-stat' : l === 'spectral' ? 'tag-spec' : l === 'clustering' ? 'tag-clust' : 'tag-art';
            return `<tr style="border-left: 3px solid ${confirmed ? '#ef4444' : '#3b82f6'}">
            <td>${s.index}</td>
            <td><strong>${riskPct}%</strong></td>
            <td><span class="tag" style="background: ${catColor}22; color: ${catColor}; border: 1px solid ${catColor}44">${riskCat}</span></td>
            <td>${s.n_layers}/6</td>
            <td>${(s.layers || []).map(l => `<span class="tag ${layerCls(l)}">${l}</span>`).join('')}</td>
            <td>${s.label ?? '–'}</td>
          </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>` : ''}

    ${profile_info ? `
    <div class="section">
      <h2>Domain Profile: ${profile_info.icon || ''} ${profile_info.name}</h2>
      <p style="color: #94a3b8; font-size: 13px; margin-bottom: 12px;">${profile_info.description || ''}</p>
      <div class="stat"><div class="val">${profile_info.z_threshold}</div><div class="lbl">Z-Threshold</div></div>
      <div class="stat"><div class="val">${profile_info.iqr_multiplier}×</div><div class="lbl">IQR Multiplier</div></div>
      <div class="stat"><div class="val">${profile_info.n_clusters}</div><div class="lbl">Clusters</div></div>
      <div class="stat"><div class="val">${(profile_info.purity_threshold * 100).toFixed(0)}%</div><div class="lbl">Min Purity</div></div>
      <p style="color: #64748b; font-size: 11px; margin-top: 12px;">Regulatory: ${profile_info.regulatory_context || 'N/A'}</p>
    </div>` : ''}

    <div class="footer">
      <p>AI PoisonGuard v1.0 — Educational Prototype — NMIMS INNOVATHON 2026</p>
      <p style="margin-top: 4px;">⚠ Not a certified security tool. Real compliance requires CERT-In empanelment, RBI ML Governance alignment, and DPDP Act compliance.</p>
    </div>
  </div>
</body>
</html>`;

    return html;
  };

  const handleDownload = () => {
    const html = generateReportHTML();
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `poisonguard_risk_report_${new Date().toISOString().slice(0, 10)}.html`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getRiskColor = () => {
    if (results.risk_level === 'LOW') return 'var(--accent-green)';
    if (results.risk_level === 'MEDIUM') return 'var(--accent-yellow)';
    if (results.risk_level === 'HIGH') return 'var(--accent-orange)';
    return 'var(--accent-red)';
  };

  return (
    <button
      onClick={handleDownload}
      className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold transition-all hover:scale-[1.02]"
      style={{
        background: `${getRiskColor()}15`,
        border: `1px solid ${getRiskColor()}40`,
        color: getRiskColor(),
        cursor: 'pointer',
      }}
    >
      <FileDown size={16} />
      Download Risk Report
    </button>
  );
}

export default RiskReportDownload;
