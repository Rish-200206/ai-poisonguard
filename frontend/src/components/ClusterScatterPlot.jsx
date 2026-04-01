import React, { useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

function ClusterScatterPlot({ scatterData }) {
  if (!scatterData || scatterData.length === 0) return null;

  const clusterColors = [
    '#3b82f6', '#8b5cf6', '#06b6d4', '#22c55e', '#eab308',
    '#f97316', '#ec4899', '#14b8a6', '#a855f7', '#6366f1',
  ];

  // Three-tier classification from ensemble voting
  const cleanData = useMemo(
    () => scatterData.filter(d => !d.is_poisoned && !d.is_warning),
    [scatterData]
  );
  const warningData = useMemo(
    () => scatterData.filter(d => d.is_warning && !d.is_poisoned),
    [scatterData]
  );
  const poisonedData = useMemo(
    () => scatterData.filter(d => d.is_poisoned),
    [scatterData]
  );

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload || !payload[0]) return null;
    const d = payload[0].payload;

    const getRiskLabel = () => {
      if (d.is_poisoned) return { text: '⚠ Suspected Poisoned', color: 'var(--accent-red)' };
      if (d.is_warning) return { text: '⚡ Single-Layer Warning', color: '#fbbf24' };
      return { text: '✓ Clean', color: 'var(--accent-green)' };
    };

    const risk = getRiskLabel();

    return (
      <div style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        padding: '10px 14px',
        fontSize: '12px',
        lineHeight: '1.6',
      }}>
        <p style={{ color: risk.color, fontWeight: 600 }}>{risk.text}</p>
        <p style={{ color: 'var(--text-secondary)' }}>Sample #{d.index}</p>
        <p style={{ color: 'var(--text-secondary)' }}>Cluster: {d.cluster}</p>
        <p style={{ color: 'var(--text-secondary)' }}>Composite Score: {d.score.toFixed(3)}</p>
        {d.ensemble_risk_score != null && (
          <p style={{ color: 'var(--text-secondary)' }}>
            Ensemble Risk: {d.ensemble_risk_score}% ({d.risk_category})
          </p>
        )}
        {d.vote_count != null && (
          <p style={{ color: 'var(--text-secondary)' }}>Layer Votes: {d.vote_count}/6</p>
        )}
        {d.label !== null && (
          <p style={{ color: 'var(--text-secondary)' }}>Label: {d.label}</p>
        )}
      </div>
    );
  };

  return (
    <div className="glass-card p-6 animate-fadeIn stagger-2">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold uppercase tracking-wider"
          style={{ color: 'var(--text-secondary)' }}>
          UMAP Cluster Projection — Ensemble Voting
        </h3>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5">
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--accent-green)' }} />
            <span style={{ color: 'var(--text-muted)' }}>Clean ({cleanData.length})</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#fbbf24' }} />
            <span style={{ color: 'var(--text-muted)' }}>Warning ({warningData.length})</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--accent-red)' }} />
            <span style={{ color: 'var(--text-muted)' }}>Poisoned ({poisonedData.length})</span>
          </div>
        </div>
      </div>

      <div style={{ width: '100%', height: 380 }}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              type="number" dataKey="x" name="UMAP-1"
              tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
              axisLine={{ stroke: 'var(--border)' }}
              label={{ value: 'UMAP Component 1', position: 'bottom', fill: 'var(--text-muted)', fontSize: 11 }}
            />
            <YAxis
              type="number" dataKey="y" name="UMAP-2"
              tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
              axisLine={{ stroke: 'var(--border)' }}
              label={{ value: 'UMAP 2', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 11 }}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />

            {/* Clean samples */}
            <Scatter name="Clean" data={cleanData} fill="var(--accent-green)">
              {cleanData.map((entry, i) => (
                <Cell
                  key={`clean-${i}`}
                  fill={clusterColors[entry.cluster % clusterColors.length]}
                  fillOpacity={0.4}
                  r={3}
                />
              ))}
            </Scatter>

            {/* Warning samples (single-layer flag) */}
            <Scatter name="Warning" data={warningData} fill="#fbbf24">
              {warningData.map((entry, i) => (
                <Cell
                  key={`warn-${i}`}
                  fill="#fbbf24"
                  fillOpacity={0.7}
                  r={4}
                  stroke="#fcd34d"
                  strokeWidth={1}
                />
              ))}
            </Scatter>

            {/* Poisoned samples (ensemble confirmed) */}
            <Scatter name="Poisoned" data={poisonedData} fill="var(--accent-red)">
              {poisonedData.map((entry, i) => (
                <Cell
                  key={`poison-${i}`}
                  fill="#ef4444"
                  fillOpacity={0.9}
                  r={5}
                  stroke="#fca5a5"
                  strokeWidth={1}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default ClusterScatterPlot;
