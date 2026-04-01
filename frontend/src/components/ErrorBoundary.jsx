import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('PoisonGuard Error:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center"
          style={{ background: 'var(--gradient-bg)' }}>
          <div className="glass-card p-10 max-w-md text-center animate-fadeIn">
            <div className="inline-flex p-4 rounded-full mb-5"
              style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
              <AlertTriangle size={36} style={{ color: 'var(--accent-red)' }} />
            </div>
            <h2 className="text-xl font-bold mb-3" style={{ color: 'var(--text-primary)' }}>
              Something went wrong
            </h2>
            <p className="text-sm mb-6" style={{ color: 'var(--text-muted)', lineHeight: 1.6 }}>
              An unexpected error occurred in the detection dashboard.
              This won't affect your data or backend analysis.
            </p>
            <pre className="text-xs p-3 rounded-lg mb-6 text-left overflow-auto"
              style={{
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border)',
                color: 'var(--accent-red)',
                maxHeight: '120px',
              }}>
              {this.state.error?.message || 'Unknown error'}
            </pre>
            <button
              onClick={this.handleReset}
              className="btn-primary inline-flex items-center gap-2"
            >
              <RefreshCw size={16} />
              Reload Dashboard
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
