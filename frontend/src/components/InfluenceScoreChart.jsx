import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

function InfluenceScoreChart({ influenceScores }) {
  if (!influenceScores || influenceScores.length === 0) return null;

  const chartData = influenceScores.map((d, i) => ({
    name: `#${d.index}`,
    score: parseFloat(d.score.toFixed(4)),
    index: d.index,
    is_flagged: d.is_flagged,
    label: d.label,
  }));

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
        <p style={{ fontWeight: 600, color: d.is_flagged ? 'var(--accent-red)' : 'var(--accent-blue)' }}>
          Sample {d.name}
        </p>
        <p style={{ color: 'var(--text-secondary)' }}>Influence Score: {d.score.toFixed(4)}</p>
        <p style={{ color: 'var(--text-secondary)' }}>
          Status: {d.is_flagged ? '⚠ Flagged' : '✓ Clean'}
        </p>
        {d.label !== null && (
          <p style={{ color: 'var(--text-secondary)' }}>Label: {d.label}</p>
        )}
      </div>
    );
  };

  return (
    <div className="glass-card p-6 animate-fadeIn stagger-3">
      <h3 className="text-sm font-semibold uppercase tracking-wider mb-1"
        style={{ color: 'var(--text-secondary)' }}>
        Influence Score Ranking
      </h3>
      <p className="text-xs mb-4" style={{ color: 'var(--text-muted)' }}>
        Top 20 most suspicious samples ranked by composite influence score
      </p>

      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <BarChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="name"
              tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
              axisLine={{ stroke: 'var(--border)' }}
              interval={0}
              angle={-45}
              textAnchor="end"
              height={50}
            />
            <YAxis
              tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
              axisLine={{ stroke: 'var(--border)' }}
              domain={[0, 1]}
              label={{ value: 'Score', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 11 }}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(59, 130, 246, 0.05)' }} />
            <Bar dataKey="score" radius={[3, 3, 0, 0]} maxBarSize={20}>
              {chartData.map((entry, i) => (
                <Cell
                  key={i}
                  fill={entry.is_flagged ? '#ef4444' : '#3b82f6'}
                  fillOpacity={0.8}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default InfluenceScoreChart;
