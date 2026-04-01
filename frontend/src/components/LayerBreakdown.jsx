import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { TrendingUp, Layers, GitBranch, Shield } from 'lucide-react';

function LayerBreakdown({ statistical, spectral, clustering, art, influence, backdoor }) {
  const layers = [
    {
      key: 'statistical',
      name: 'Layer 1 — Statistical Analysis',
      subtitle: 'Z-score + IQR outlier detection',
      icon: <TrendingUp size={15} />,
      color: '#38bdf8',
      gradient: 'linear-gradient(135deg, #38bdf8, #3b82f6)',
      data: statistical,
    },
    {
      key: 'spectral',
      name: 'Layer 2 — Spectral Signatures',
      subtitle: 'SVD-based spectral analysis',
      icon: <Layers size={15} />,
      color: '#818cf8',
      gradient: 'linear-gradient(135deg, #818cf8, #a78bfa)',
      data: spectral,
    },
    {
      key: 'clustering',
      name: 'Layer 3 — Activation Clustering',
      subtitle: 'PyTorch MLP + UMAP + KMeans',
      icon: <GitBranch size={15} />,
      color: '#22d3ee',
      gradient: 'linear-gradient(135deg, #22d3ee, #06b6d4)',
      data: clustering,
    },
  ];

  const summaryData = layers.map(l => ({
    name: l.key.charAt(0).toUpperCase() + l.key.slice(1),
    flagged: l.data?.n_flagged || 0,
    color: l.color,
  }));

  if (art) {
    summaryData.push({
      name: 'ART',
      flagged: art.n_flagged || 0,
      color: '#fb923c',
    });
  }
  if (influence) {
    summaryData.push({
      name: 'Influence',
      flagged: influence.n_flagged || 0,
      color: '#f472b6',
    });
  }
  if (backdoor) {
    summaryData.push({
      name: 'Backdoor',
      flagged: backdoor.n_flagged || 0,
      color: '#a78bfa',
    });
  }

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.[0]) return null;
    return (
      <div style={{
        background: 'rgba(12, 18, 34, 0.95)',
        backdropFilter: 'blur(12px)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-md)',
        padding: '10px 14px',
        fontSize: '12px',
      }}>
        <p style={{ fontWeight: 600, color: payload[0].payload.color, marginBottom: '2px' }}>
          {label}
        </p>
        <p style={{ color: 'var(--text-secondary)' }}>
          {payload[0].value} samples flagged
        </p>
      </div>
    );
  };

  return (
    <div className="glass-card p-6 animate-fadeIn stagger-2">
      <div className="flex items-center gap-2 mb-5">
        <Shield size={15} style={{ color: 'var(--accent-indigo)' }} />
        <h3 style={{
          fontSize: '11px',
          fontWeight: 700,
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
          color: 'var(--text-muted)',
        }}>
          Detection Layer Breakdown
        </h3>
      </div>

      {/* Summary chart */}
      <div style={{ width: '100%', height: 150, marginBottom: '20px' }}>
        <ResponsiveContainer>
          <BarChart data={summaryData} layout="vertical"
            margin={{ top: 0, right: 20, bottom: 0, left: 70 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(56, 97, 180, 0.08)" horizontal={false} />
            <XAxis type="number"
              tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'Inter' }}
              axisLine={{ stroke: 'var(--border-subtle)' }}
              tickLine={false}
            />
            <YAxis type="category" dataKey="name"
              tick={{ fill: 'var(--text-secondary)', fontSize: 11, fontWeight: 600, fontFamily: 'Inter' }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(56, 189, 248, 0.03)' }} />
            <Bar dataKey="flagged" radius={[0, 6, 6, 0]} maxBarSize={18}>
              {summaryData.map((entry, i) => (
                <Cell key={i} fill={entry.color} fillOpacity={0.75} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Layer cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {layers.map((layer) => {
          if (!layer.data) return null;
          const d = layer.data;
          return (
            <div key={layer.key}
              style={{
                padding: '14px 16px',
                borderRadius: 'var(--radius-md)',
                background: `${layer.color}06`,
                border: `1px solid ${layer.color}18`,
                position: 'relative',
                overflow: 'hidden',
              }}>
              {/* Left accent bar */}
              <div style={{
                position: 'absolute',
                left: 0,
                top: 0,
                bottom: 0,
                width: '3px',
                background: layer.gradient,
                borderRadius: '3px 0 0 3px',
              }} />

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', paddingLeft: '8px' }}>
                <span style={{ color: layer.color }}>{layer.icon}</span>
                <span style={{
                  fontSize: '13px',
                  fontWeight: 700,
                  color: 'var(--text-primary)',
                  letterSpacing: '-0.01em',
                }}>
                  {layer.name}
                </span>
              </div>
              <p style={{
                fontSize: '11px',
                marginBottom: '10px',
                color: 'var(--text-muted)',
                paddingLeft: '8px',
              }}>
                {layer.subtitle}
              </p>
              <div style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '16px',
                fontSize: '12px',
                paddingLeft: '8px',
              }}>
                <div>
                  <span style={{ color: 'var(--text-muted)' }}>Flagged: </span>
                  <span style={{
                    fontWeight: 700,
                    color: d.n_flagged > 0 ? 'var(--accent-red)' : 'var(--accent-green)',
                    fontFamily: "'JetBrains Mono', monospace",
                  }}>
                    {d.n_flagged} ({(d.flagged_ratio * 100).toFixed(1)}%)
                  </span>
                </div>
                {layer.key === 'statistical' && d.label_analysis && (
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Labels: </span>
                    <span style={{
                      fontWeight: 700,
                      color: d.label_analysis.is_anomalous ? 'var(--accent-red)' : 'var(--accent-green)',
                    }}>
                      {d.label_analysis.is_anomalous ? 'Anomalous' : 'Normal'}
                    </span>
                  </div>
                )}
                {layer.key === 'spectral' && d.singular_value_analysis && (
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>SV Ratio: </span>
                    <span style={{
                      fontWeight: 700,
                      fontFamily: "'JetBrains Mono', monospace",
                      color: d.singular_value_analysis.is_suspicious ? 'var(--accent-red)' : 'var(--accent-green)',
                    }}>
                      {d.singular_value_analysis.sv_ratio_1_2?.toFixed(2)}x
                    </span>
                  </div>
                )}
                {layer.key === 'clustering' && (
                  <>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Silhouette: </span>
                      <span style={{
                        fontWeight: 700,
                        color: 'var(--text-primary)',
                        fontFamily: "'JetBrains Mono', monospace",
                      }}>
                        {d.silhouette_score?.toFixed(3)}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-muted)' }}>Shadow: </span>
                      <span style={{
                        fontWeight: 600,
                        color: 'var(--accent-cyan)',
                        fontSize: '11px',
                      }}>
                        {d.shadow_model || 'N/A'}
                      </span>
                    </div>
                  </>
                )}
              </div>
            </div>
          );
        })}

        {/* ART layer */}
        {art && (
          <div style={{
            padding: '14px 16px',
            borderRadius: 'var(--radius-md)',
            background: 'rgba(251, 146, 60, 0.04)',
            border: '1px solid rgba(251, 146, 60, 0.12)',
            position: 'relative',
            overflow: 'hidden',
          }}>
            <div style={{
              position: 'absolute',
              left: 0,
              top: 0,
              bottom: 0,
              width: '3px',
              background: 'linear-gradient(135deg, #fb923c, #f97316)',
              borderRadius: '3px 0 0 3px',
            }} />
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginBottom: '8px',
              paddingLeft: '8px',
            }}>
              <span style={{ fontSize: '15px' }}>🛡️</span>
              <span style={{
                fontSize: '13px',
                fontWeight: 700,
                color: 'var(--text-primary)',
              }}>
                Layer 4 — IBM ART Validation
              </span>
              <span className={`tag ${art.art_available ? 'tag-low' : 'tag-medium'}`}
                style={{ fontSize: '9px', padding: '2px 8px' }}>
                {art.art_available ? 'ART Active' : 'Fallback'}
              </span>
            </div>
            <div style={{ fontSize: '12px', paddingLeft: '8px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Flagged: </span>
              <span style={{
                fontWeight: 700,
                fontFamily: "'JetBrains Mono', monospace",
                color: art.n_flagged > 0 ? 'var(--accent-red)' : 'var(--accent-green)',
              }}>
                {art.n_flagged} ({(art.flagged_ratio * 100).toFixed(1)}%)
              </span>
            </div>
          </div>
        )}

        {/* Influence layer */}
        {influence && (
          <div style={{
            padding: '14px 16px',
            borderRadius: 'var(--radius-md)',
            background: 'rgba(244, 114, 182, 0.04)',
            border: '1px solid rgba(244, 114, 182, 0.12)',
            position: 'relative',
            overflow: 'hidden',
          }}>
            <div style={{
              position: 'absolute',
              left: 0,
              top: 0,
              bottom: 0,
              width: '3px',
              background: 'linear-gradient(135deg, #f472b6, #ec4899)',
              borderRadius: '3px 0 0 3px',
            }} />
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginBottom: '8px',
              paddingLeft: '8px',
            }}>
              <span style={{ fontSize: '15px' }}>📈</span>
              <span style={{
                fontSize: '13px',
                fontWeight: 700,
                color: 'var(--text-primary)',
              }}>
                Layer 5 — Influence Function Analysis
              </span>
              <span className={`tag ${influence.method === 'tracin' ? 'tag-low' : 'tag-medium'}`}
                style={{ fontSize: '9px', padding: '2px 8px' }}>
                {influence.method === 'tracin' ? 'TracIn' : influence.method === 'loo' ? 'LOO' : 'N/A'}
              </span>
            </div>
            <div style={{ fontSize: '12px', paddingLeft: '8px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Flagged: </span>
              <span style={{
                fontWeight: 700,
                fontFamily: "'JetBrains Mono', monospace",
                color: influence.n_flagged > 0 ? 'var(--accent-red)' : 'var(--accent-green)',
              }}>
                {influence.n_flagged} ({(influence.flagged_ratio * 100).toFixed(1)}%)
              </span>
            </div>
          </div>
        )}

        {/* Backdoor layer */}
        {backdoor && (
          <div style={{
            padding: '14px 16px',
            borderRadius: 'var(--radius-md)',
            background: 'rgba(167, 139, 250, 0.04)',
            border: '1px solid rgba(167, 139, 250, 0.12)',
            position: 'relative',
            overflow: 'hidden',
          }}>
            <div style={{
              position: 'absolute',
              left: 0,
              top: 0,
              bottom: 0,
              width: '3px',
              background: 'linear-gradient(135deg, #a78bfa, #8b5cf6)',
              borderRadius: '3px 0 0 3px',
            }} />
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginBottom: '8px',
              paddingLeft: '8px',
            }}>
              <span style={{ fontSize: '15px' }}>🔓</span>
              <span style={{
                fontSize: '13px',
                fontWeight: 700,
                color: 'var(--text-primary)',
              }}>
                Layer 6 — Backdoor Trigger Scanner
              </span>
              {backdoor.n_trigger_patterns_found > 0 && (
                <span className="tag tag-critical"
                  style={{ fontSize: '9px', padding: '2px 8px' }}>
                  {backdoor.n_trigger_patterns_found} patterns
                </span>
              )}
            </div>
            <div style={{ fontSize: '12px', paddingLeft: '8px' }}>
              <span style={{ color: 'var(--text-muted)' }}>Flagged: </span>
              <span style={{
                fontWeight: 700,
                fontFamily: "'JetBrains Mono', monospace",
                color: backdoor.n_flagged > 0 ? 'var(--accent-red)' : 'var(--accent-green)',
              }}>
                {backdoor.n_flagged} ({(backdoor.flagged_ratio * 100).toFixed(1)}%)
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default LayerBreakdown;
