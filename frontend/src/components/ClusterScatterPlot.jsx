import React, { useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

function ClusterScatterPlot({ scatterData }) {
  if (!scatterData || scatterData.length === 0) return null;

  const clusterColors = [
    '#3b82f6', '#8b5cf6', '#06b6d4', '#22c55e', '#eab308',
    '#f97316', '#ec4899', '#14b8a6', '#a855f7', '#6366f1',
  ];

  const cleanData = useMemo(() => scatterData.filter(d => !d.is_poisoned), [scatterData]);
  const poisonedData = useMemo(() => scatterData.filter(d => d.is_poisoned), [scatterData]);

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload || !payload[0]) return null;
    const d = payload[0].payload;
    return (
      <div style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-sm)',
        padding: '10px 14px',
        fontSize: '12px',
        lineHeight: '1.6',
      }}>
        <p style={{ color: d.is_poisoned ? 'var(--accent-red)' : 'var(--accent-green)', fontWeight: 600 }}>
          {d.is_poisoned ? '⚠ Suspected Poisoned' : '✓ Clean'}
        </p>
        <p style={{ color: 'var(--text-secondary)' }}>Sample #{d.index}</p>
        <p style={{ color: 'var(--text-secondary)' }}>Cluster: {d.cluster}</p>
        <p style={{ color: 'var(--text-secondary)' }}>Score: {d.score.toFixed(3)}</p>
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
          UMAP Cluster Projection
        </h3>
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5">
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--accent-green)' }} />
            <span style={{ color: 'var(--text-muted)' }}>Clean ({cleanData.length})</span>
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
                  fillOpacity={0.5}
                  r={3}
                />
              ))}
            </Scatter>

            {/* Poisoned samples */}
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
