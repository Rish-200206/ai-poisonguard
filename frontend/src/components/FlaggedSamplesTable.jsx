import React, { useState, useMemo } from 'react';
import { Download, ChevronUp, ChevronDown } from 'lucide-react';
import Papa from 'papaparse';

function FlaggedSamplesTable({ flaggedSamples }) {
  const [sortKey, setSortKey] = useState('score');
  const [sortDir, setSortDir] = useState('desc');
  const [page, setPage] = useState(0);
  const perPage = 15;

  if (!flaggedSamples || flaggedSamples.length === 0) return null;

  const sorted = useMemo(() => {
    return [...flaggedSamples].sort((a, b) => {
      let av = a[sortKey], bv = b[sortKey];
      if (sortKey === 'n_layers') { av = a.n_layers; bv = b.n_layers; }
      if (sortDir === 'asc') return av > bv ? 1 : -1;
      return av < bv ? 1 : -1;
    });
  }, [flaggedSamples, sortKey, sortDir]);

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
        score: s.score.toFixed(4),
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
            Suspected Poison Samples
          </h3>
          <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
            {flaggedSamples.length} samples flagged across detection layers
          </p>
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

      <div style={{ overflowX: 'auto', maxHeight: '400px', overflowY: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              <th onClick={() => toggleSort('index')} style={{ cursor: 'pointer' }}>
                Index <SortIcon col="index" />
              </th>
              <th onClick={() => toggleSort('score')} style={{ cursor: 'pointer' }}>
                Score <SortIcon col="score" />
              </th>
              <th onClick={() => toggleSort('n_layers')} style={{ cursor: 'pointer' }}>
                Layers <SortIcon col="n_layers" />
              </th>
              <th>Detection Sources</th>
              <th>Risk Reason</th>
              <th>Label</th>
            </tr>
          </thead>
          <tbody>
            {paginated.map((sample) => (
              <tr key={sample.index}>
                <td style={{ fontFamily: 'monospace', fontSize: '12px' }}>
                  {sample.index}
                </td>
                <td>
                  <div className="flex items-center gap-2">
                    <div style={{
                      width: '40px',
                      height: '4px',
                      borderRadius: '2px',
                      background: 'var(--border)',
                      overflow: 'hidden',
                    }}>
                      <div style={{
                        width: `${sample.score * 100}%`,
                        height: '100%',
                        borderRadius: '2px',
                        background: sample.score > 0.7 ? 'var(--accent-red)'
                          : sample.score > 0.4 ? 'var(--accent-orange)' : 'var(--accent-yellow)',
                      }} />
                    </div>
                    <span style={{ fontFamily: 'monospace', fontSize: '12px' }}>
                      {sample.score.toFixed(3)}
                    </span>
                  </div>
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
                    maxWidth: '280px',
                  }}>
                    {sample.risk_reason || '—'}
                  </span>
                </td>
                <td style={{ fontFamily: 'monospace', fontSize: '12px' }}>
                  {sample.label ?? '–'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-4 text-xs"
          style={{ color: 'var(--text-muted)' }}>
          <span>
            Page {page + 1} of {totalPages}
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
