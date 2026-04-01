import React from 'react';
import { Shield, AlertTriangle, BookOpen, Settings2 } from 'lucide-react';

function DomainProfileCard({ profileInfo, modelInfo }) {
  if (!profileInfo) return null;

  const getRiskBadge = () => {
    const cat = profileInfo.risk_category;
    if (cat === 'HIGH') return 'tag-high';
    if (cat === 'MEDIUM') return 'tag-medium';
    return 'tag-low';
  };

  return (
    <div className="glass-card p-6 animate-fadeIn stagger-3">
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
        <span style={{ fontSize: '28px' }}>{profileInfo.icon}</span>
        <div>
          <h3 style={{
            fontSize: '15px',
            fontWeight: 800,
            color: 'var(--text-bright)',
            letterSpacing: '-0.01em',
          }}>
            {profileInfo.name}
          </h3>
          <span className={`tag ${getRiskBadge()}`} style={{ fontSize: '9px', marginTop: '4px' }}>
            {profileInfo.risk_category} RISK DOMAIN
          </span>
        </div>
      </div>

      <p style={{
        fontSize: '12px',
        marginBottom: '16px',
        lineHeight: 1.6,
        color: 'var(--text-muted)',
      }}>
        {profileInfo.description}
      </p>

      {/* Regulatory */}
      <div style={{
        padding: '12px',
        borderRadius: 'var(--radius-md)',
        background: 'rgba(99, 102, 241, 0.04)',
        border: '1px solid rgba(99, 102, 241, 0.1)',
        marginBottom: '16px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
          <BookOpen size={12} style={{ color: 'var(--accent-indigo)' }} />
          <span style={{ fontSize: '11px', fontWeight: 700, color: 'var(--accent-indigo)', letterSpacing: '0.03em' }}>
            Regulatory Context
          </span>
        </div>
        <p style={{ fontSize: '11px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
          {profileInfo.regulatory_context}
        </p>
      </div>

      {/* Thresholds */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '10px' }}>
        <Settings2 size={12} style={{ color: 'var(--text-muted)' }} />
        <span style={{
          fontSize: '10px',
          fontWeight: 700,
          color: 'var(--text-muted)',
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
        }}>
          Detection Thresholds
        </span>
      </div>
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '6px',
        marginBottom: '16px',
      }}>
        {[
          { label: 'Z-Score', value: profileInfo.z_threshold },
          { label: 'IQR ×', value: profileInfo.iqr_multiplier },
          { label: 'Clusters', value: profileInfo.n_clusters },
          { label: 'Min Purity', value: `${(profileInfo.purity_threshold * 100).toFixed(0)}%` },
        ].map(({ label, value }) => (
          <div key={label} style={{
            padding: '10px 12px',
            borderRadius: 'var(--radius-sm)',
            background: 'rgba(12, 18, 34, 0.5)',
            border: '1px solid var(--border-subtle)',
            textAlign: 'center',
          }}>
            <p style={{
              fontSize: '10px',
              color: 'var(--text-muted)',
              fontWeight: 600,
              letterSpacing: '0.04em',
              textTransform: 'uppercase',
            }}>{label}</p>
            <p style={{
              fontSize: '15px',
              fontWeight: 800,
              color: 'var(--text-primary)',
              fontFamily: "'JetBrains Mono', monospace",
              marginTop: '2px',
            }}>{value}</p>
          </div>
        ))}
      </div>

      {/* Model info */}
      {modelInfo && !modelInfo.error && (
        <div style={{
          padding: '12px',
          borderRadius: 'var(--radius-md)',
          background: 'rgba(167, 139, 250, 0.04)',
          border: '1px solid rgba(167, 139, 250, 0.12)',
          marginBottom: '16px',
        }}>
          <p style={{ fontSize: '11px', fontWeight: 700, color: 'var(--accent-purple)', marginBottom: '4px' }}>
            Uploaded Model
          </p>
          <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            {modelInfo.filename} — {modelInfo.type}
          </p>
        </div>
      )}

      {/* Disclaimer */}
      <div style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '8px',
        padding: '10px 12px',
        borderRadius: 'var(--radius-sm)',
        background: 'rgba(251, 191, 36, 0.04)',
        border: '1px solid rgba(251, 191, 36, 0.1)',
      }}>
        <AlertTriangle size={12} style={{
          color: 'var(--accent-yellow)',
          marginTop: '2px',
          flexShrink: 0,
        }} />
        <p style={{ fontSize: '10px', color: 'var(--text-muted)', lineHeight: 1.5 }}>
          Educational prototype. Production deployment requires CERT-In empanelment
          and RBI ML Governance compliance.
        </p>
      </div>
    </div>
  );
}

export default DomainProfileCard;
