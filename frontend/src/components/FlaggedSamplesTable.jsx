import React, { useState, useMemo } from 'react';
import { Download, ChevronUp, ChevronDown, ShieldAlert, ShieldCheck } from 'lucide-react';
import Papa from 'papaparse';

function FlaggedSamplesTable({ flaggedSamples }) {
  const [sortKey, setSortKey] = useState('ensemble_risk_score');
  const [sortDir, setSortDir] = useState('desc');
  const [page, setPage] = useState(0);
  const [filterMode, setFilterMode] = useState('all'); // 'all' | 'confirmed' | 'warnings'
  const perPage = 15;

  if (!flaggedSamples || flaggedSamples.length === 0) return null;

  const confirmedCount = flaggedSamples.filter(s => s.is_confirmed_poison).length;
  const warningCount = flaggedSamples.filter(s => !s.is_confirmed_poison).length;

  const filtered = useMemo(() => {
    if (filterMode === 'confirmed') return flaggedSamples.filter(s => s.is_confirmed_poison);
    if (filterMode === 'warnings') return flaggedSamples.filter(s => !s.is_confirmed_poison);
    return flaggedSamples;
  }, [flaggedSamples, filterMode]);

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      let av = a[sortKey] ?? 0, bv = b[sortKey] ?? 0;
      if (sortDir === 'asc') return av > bv ? 1 : -1;
      return av < bv ? 1 : -1;
    });
  }, [filtered, sortKey, sortDir]);

  const paginated = sorted.slice(page * perPage, (page + 1) * perPage);
  const totalPages = Math.ceil(sorted.length / perPage);

  const toggleSort = (key) => {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  const SortIcon = ({ col }) => {
    if (sortKey !== col) return null;
    return sortDir === 'asc'
      ? <ChevronUp size={12} className="inline ml-0.5" />
      : <ChevronDown size={12} className="inline ml-0.5" />;
  };

  const getRiskCategoryStyle = (category) => {
    switch (category) {
      case 'Critical':
        return { background: 'rgba(239, 68, 68, 0.15)', color: '#f87171', border: '1px solid rgba(239, 68, 68, 0.3)' };
      case 'High Risk':
        return { background: 'rgba(249, 115, 22, 0.15)', color: '#fb923c', border: '1px solid rgba(249, 115, 22, 0.3)' };
      case 'Compromised':
        return { background: 'rgba(234, 179, 8, 0.15)', color: '#fbbf24', border: '1px solid rgba(234, 179, 8, 0.3)' };
      case 'Warning':
        return { background: 'rgba(59, 130, 246, 0.12)', color: '#60a5fa', border: '1px solid rgba(59, 130, 246, 0.25)' };
      default:
        return { background: 'rgba(34, 197, 94, 0.12)', color: '#4ade80', border: '1px solid rgba(34, 197, 94, 0.25)' };
    }
  };

  const handleDownload = () => {
    const rows = flaggedSamples.map(s => {
      const featureCols = {};
      if (s.features) {
        Object.entries(s.features).forEach(([k, v]) => {
          featureCols[k] = typeof v === 'number' ? v.toFixed(4) : v;
        });
      }
      return {
        index: s.index,
        composite_score: s.score.toFixed(4),
        ensemble_risk_score: s.ensemble_risk_score ?? '',
        risk_category: s.risk_category ?? '',
        is_confirmed_poison: s.is_confirmed_poison ? 'YES' : 'NO',
        layers: s.layers.join(', '),
        n_layers: s.n_layers,
        risk_reason: s.risk_reason || '',
        label: s.label ?? '',
        ...featureCols,
      };
    });
    const csv = Papa.unparse(rows);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'flagged_samples.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="glass-card p-6 animate-fadeIn stagger-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-wider"
            style={{ color: 'var(--text-secondary)' }}>
            Ensemble Voting Results
          </h3>
          <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
            {confirmedCount} confirmed suspicious (≥2 layers) · {warningCount} single-layer warnings
          </p>
        </div>
        <div className="flex items-center gap-3">
          {/* Filter toggles */}
          <div style={{
            display: 'inline-flex',
            gap: '2px',
            padding: '3px',
            borderRadius: 'var(--radius-md)',
            background: 'rgba(12, 18, 34, 0.6)',
            border: '1px solid var(--border)',
          }}>
            {[
              { id: 'all', label: `All (${flaggedSamples.length})` },
              { id: 'confirmed', label: `Confirmed (${confirmedCount})` },
              { id: 'warnings', label: `Warnings (${warningCount})` },
            ].map(f => (
              <button
                key={f.id}
                onClick={() => { setFilterMode(f.id); setPage(0); }}
                style={{
                  padding: '4px 10px',
                  borderRadius: '6px',
                  fontSize: '10px',
                  fontWeight: 600,
                  background: filterMode === f.id ? 'var(--gradient-primary)' : 'transparent',
                  color: filterMode === f.id ? 'white' : 'var(--text-muted)',
                  cursor: 'pointer',
                  border: 'none',
                  transition: 'all 0.2s ease',
                  letterSpacing: '0.01em',
                }}
              >
                {f.label}
              </button>
            ))}
          </div>

          <button
            onClick={handleDownload}
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold hover:opacity-80 transition-opacity"
            style={{
              background: 'rgba(34, 197, 94, 0.1)',
              border: '1px solid rgba(34, 197, 94, 0.3)',
              color: 'var(--accent-green)',
              cursor: 'pointer',
            }}
          >
            <Download size={14} />
            Export CSV
          </button>
        </div>
      </div>

      <div style={{ overflowX: 'auto', maxHeight: '450px', overflowY: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th onClick={() => toggleSort('index')} style={{ cursor: 'pointer' }}>
                Index <SortIcon col="index" />
              </th>
              <th onClick={() => toggleSort('ensemble_risk_score')} style={{ cursor: 'pointer' }}>
                Risk Score <SortIcon col="ensemble_risk_score" />
              </th>
              <th onClick={() => toggleSort('risk_category')} style={{ cursor: 'pointer' }}>
                Risk Level <SortIcon col="risk_category" />
              </th>
              <th onClick={() => toggleSort('n_layers')} style={{ cursor: 'pointer' }}>
                Votes <SortIcon col="n_layers" />
              </th>
              <th>Detection Sources</th>
              <th>Risk Reason</th>
              <th>Label</th>
            </tr>
          </thead>
          <tbody>
            {paginated.map((sample) => {
              const catStyle = getRiskCategoryStyle(sample.risk_category);
              const riskPct = sample.ensemble_risk_score ?? 0;

              return (
                <tr key={sample.index} style={{
                  opacity: sample.is_confirmed_poison ? 1 : 0.7,
                  borderLeft: sample.is_confirmed_poison
                    ? '3px solid var(--accent-red)'
                    : '3px solid rgba(59, 130, 246, 0.3)',
                }}>
                  <td style={{ fontFamily: 'monospace', fontSize: '12px' }}>
                    <div className="flex items-center gap-1.5">
                      {sample.is_confirmed_poison
                        ? <ShieldAlert size={13} style={{ color: 'var(--accent-red)', flexShrink: 0 }} />
                        : <ShieldCheck size={13} style={{ color: 'var(--accent-blue, #60a5fa)', flexShrink: 0, opacity: 0.6 }} />
                      }
                      {sample.index}
                    </div>
                  </td>
                  <td>
                    <div className="flex items-center gap-2">
                      <div style={{
                        width: '48px',
                        height: '5px',
                        borderRadius: '3px',
                        background: 'var(--border)',
                        overflow: 'hidden',
                      }}>
                        <div style={{
                          width: `${riskPct}%`,
                          height: '100%',
                          borderRadius: '3px',
                          background: riskPct >= 80 ? 'var(--accent-red)'
                            : riskPct >= 60 ? 'var(--accent-orange)'
                            : riskPct >= 40 ? 'var(--accent-yellow)'
                            : 'var(--accent-blue, #60a5fa)',
                          transition: 'width 0.3s ease',
                        }} />
                      </div>
                      <span style={{
                        fontFamily: 'monospace',
                        fontSize: '12px',
                        fontWeight: 700,
                        color: riskPct >= 80 ? 'var(--accent-red)'
                          : riskPct >= 60 ? 'var(--accent-orange)'
                          : riskPct >= 40 ? 'var(--accent-yellow)'
                          : 'var(--text-secondary)',
                      }}>
                        {riskPct}%
                      </span>
                    </div>
                  </td>
                  <td>
                    <span style={{
                      ...catStyle,
                      padding: '3px 10px',
                      borderRadius: '100px',
                      fontSize: '10px',
                      fontWeight: 700,
                      letterSpacing: '0.03em',
                      textTransform: 'uppercase',
                      whiteSpace: 'nowrap',
                    }}>
                      {sample.risk_category || 'Unknown'}
                    </span>
                  </td>
                  <td>
                    <span className={`tag ${
                      sample.n_layers >= 3 ? 'tag-critical'
                      : sample.n_layers >= 2 ? 'tag-high'
                      : 'tag-medium'
                    }`}>
                      {sample.n_layers}/{6}
                    </span>
                  </td>
                  <td>
                    <div className="flex flex-wrap gap-1">
                      {sample.layers.map(l => (
                        <span key={l} style={{
                          fontSize: '10px',
                          padding: '2px 6px',
                          borderRadius: '4px',
                          background: l === 'statistical' ? 'rgba(59, 130, 246, 0.15)'
                            : l === 'spectral' ? 'rgba(139, 92, 246, 0.15)'
                            : l === 'clustering' ? 'rgba(6, 182, 212, 0.15)'
                            : l === 'influence' ? 'rgba(244, 114, 182, 0.15)'
                            : l === 'backdoor' ? 'rgba(167, 139, 250, 0.15)'
                            : 'rgba(249, 115, 22, 0.15)',
                          color: l === 'statistical' ? '#60a5fa'
                            : l === 'spectral' ? '#a78bfa'
                            : l === 'clustering' ? '#22d3ee'
                            : l === 'influence' ? '#f472b6'
                            : l === 'backdoor' ? '#a78bfa'
                            : '#fb923c',
                          fontWeight: 500,
                        }}>
                          {l}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td>
                    <span style={{
                      fontSize: '11px',
                      color: 'var(--text-secondary)',
                      lineHeight: 1.4,
                      display: 'block',
                      maxWidth: '300px',
                    }}>
                      {sample.risk_reason || '—'}
                    </span>
                  </td>
                  <td style={{ fontFamily: 'monospace', fontSize: '12px' }}>
                    {sample.label ?? '–'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-4 text-xs"
          style={{ color: 'var(--text-muted)' }}>
          <span>
            Page {page + 1} of {totalPages} · {sorted.length} samples
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage(p => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-3 py-1.5 rounded-md"
              style={{
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border)',
                color: page === 0 ? 'var(--text-muted)' : 'var(--text-secondary)',
                cursor: page === 0 ? 'not-allowed' : 'pointer',
              }}
            >
              Previous
            </button>
            <button
              onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="px-3 py-1.5 rounded-md"
              style={{
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border)',
                color: page >= totalPages - 1 ? 'var(--text-muted)' : 'var(--text-secondary)',
                cursor: page >= totalPages - 1 ? 'not-allowed' : 'pointer',
              }}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default FlaggedSamplesTable;
