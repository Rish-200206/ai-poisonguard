import React, { useEffect, useState } from 'react';

function RiskScoreCard({ riskScore, riskLevel, nFlagged, nWarnings = 0, nSamples, status, ensembleThreshold = 2 }) {
  const [displayScore, setDisplayScore] = useState(0);

  useEffect(() => {
    const target = riskScore;
    const duration = 2000;
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 4); // easeOutQuart
      setDisplayScore(Math.round(target * eased));
      if (progress < 1) requestAnimationFrame(animate);
    };

    requestAnimationFrame(animate);
  }, [riskScore]);

  const getColor = () => {
    if (riskScore < 15) return '#34d399';
    if (riskScore < 40) return '#fbbf24';
    if (riskScore < 70) return '#fb923c';
    return '#f87171';
  };

  const getGlowShadow = () => {
    const c = getColor();
    return `0 0 40px ${c}20, 0 0 80px ${c}10`;
  };

  const getTagClass = () => {
    if (riskLevel === 'LOW') return 'tag-low';
    if (riskLevel === 'MEDIUM') return 'tag-medium';
    if (riskLevel === 'HIGH') return 'tag-high';
    return 'tag-critical';
  };

  const circumference = 2 * Math.PI * 68;
  const offset = circumference - (riskScore / 100) * circumference;
  const color = getColor();

  return (
    <div className="glass-card p-6 animate-fadeIn stagger-1" style={{
      textAlign: 'center',
      borderTop: `2px solid ${color}`,
      boxShadow: getGlowShadow(),
    }}>
      <h3 style={{
        fontSize: '11px',
        fontWeight: 700,
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
        color: 'var(--text-muted)',
        marginBottom: '20px',
      }}>
        Poison Risk Score
      </h3>

      {/* Gauge */}
      <div className="risk-gauge mx-auto" style={{ marginBottom: '20px' }}>
        <svg viewBox="0 0 160 160" width="180" height="180">
          {/* Outer ring glow */}
          <circle
            cx="80" cy="80" r="72"
            fill="none"
            stroke={`${color}08`}
            strokeWidth="16"
          />
          {/* Background circle */}
          <circle
            cx="80" cy="80" r="68"
            fill="none"
            stroke="rgba(56, 97, 180, 0.08)"
            strokeWidth="7"
          />
          {/* Active arc */}
          <circle
            cx="80" cy="68" r="68"
            className="risk-gauge-circle"
            stroke={color}
            strokeWidth="7"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{
              filter: `drop-shadow(0 0 8px ${color}80)`,
              transform: 'rotate(-90deg)',
              transformOrigin: '80px 68px',
            }}
          />
          {/* Score */}
          <text x="80" y="70" textAnchor="middle"
            style={{
              fill: color,
              fontSize: '38px',
              fontWeight: 900,
              fontFamily: 'Inter, sans-serif',
              letterSpacing: '-0.02em',
            }}>
            {displayScore}
          </text>
          <text x="80" y="92" textAnchor="middle"
            style={{ fill: 'var(--text-muted)', fontSize: '11px', fontWeight: 500 }}>
            / 100
          </text>
        </svg>
      </div>

      {/* Badge */}
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '16px' }}>
        <span className={`tag ${getTagClass()}`} style={{
          padding: '5px 16px',
          fontSize: '11px',
        }}>
          {riskLevel} RISK
        </span>
      </div>

      {/* Ensemble voting note */}
      <p style={{
        fontSize: '10px',
        color: 'var(--text-muted)',
        marginBottom: '16px',
        opacity: 0.7,
        lineHeight: 1.4,
      }}>
        Ensemble Voting: ≥{ensembleThreshold} layer agreement required
      </p>

      {/* Stats */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr 1fr 1fr',
        gap: '6px',
      }}>
        {[
          { label: 'Status', value: status, vColor: color },
          { label: 'Confirmed', value: nFlagged.toLocaleString(), vColor: 'var(--accent-red)' },
          { label: 'Warnings', value: nWarnings.toLocaleString(), vColor: '#fbbf24' },
          { label: 'Total', value: nSamples.toLocaleString(), vColor: 'var(--text-primary)' },
        ].map(({ label, value, vColor }) => (
          <div key={label} style={{
            padding: '10px 6px',
            borderRadius: 'var(--radius-sm)',
            background: 'rgba(12, 18, 34, 0.5)',
            border: '1px solid var(--border-subtle)',
          }}>
            <p style={{
              fontSize: '9px',
              color: 'var(--text-muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              fontWeight: 600,
              marginBottom: '2px',
            }}>{label}</p>
            <p style={{
              fontWeight: 700,
              color: vColor,
              fontSize: '13px',
              textTransform: label === 'Status' ? 'capitalize' : 'none',
            }}>{value}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default RiskScoreCard;
