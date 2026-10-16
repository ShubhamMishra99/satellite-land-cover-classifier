import React, { useEffect, useState } from 'react';
import { Globe, Satellite, Menu, X, Cpu, CheckCircle2, AlertCircle } from 'lucide-react';

export const Navbar: React.FC = () => {
  const [isBackendHealthy, setIsBackendHealthy] = useState<boolean | null>(null);
  const [deviceInfo, setDeviceInfo] = useState<string>('CPU');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  // Poll health endpoint once on mount and every 25 seconds
  useEffect(() => {
    let isMounted = true;

    const checkHealth = async () => {
      try {
        const res = await fetch('/health');
        if (res.ok) {
          const data = await res.json();
          if (isMounted) {
            setIsBackendHealthy(data.status === 'healthy');
            if (data.device) setDeviceInfo(data.device.toUpperCase());
          }
        } else {
          if (isMounted) setIsBackendHealthy(false);
        }
      } catch {
        if (isMounted) setIsBackendHealthy(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 25000);

    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);

    return () => {
      isMounted = false;
      clearInterval(interval);
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'bg-space/85 backdrop-blur-md border-b border-white/10 shadow-lg py-3'
          : 'bg-transparent py-5'
      }`}
      style={{
        backgroundColor: scrolled ? 'rgba(3, 7, 18, 0.88)' : 'transparent',
        backdropFilter: scrolled ? 'blur(16px)' : 'none',
        borderBottom: scrolled ? '1px solid rgba(255, 255, 255, 0.08)' : 'none',
      }}
    >
      <div className="container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        {/* Brand */}
        <a
          href="#hero"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            textDecoration: 'none',
            color: '#FFFFFF',
          }}
        >
          <div
            style={{
              width: '38px',
              height: '38px',
              borderRadius: '10px',
              background: 'linear-gradient(135deg, rgba(0, 240, 255, 0.2), rgba(2, 132, 199, 0.4))',
              border: '1px solid rgba(0, 240, 255, 0.4)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 15px rgba(0, 240, 255, 0.3)',
            }}
          >
            <Satellite size={20} color="#00F0FF" />
          </div>
          <div>
            <span
              style={{
                fontFamily: 'var(--font-display)',
                fontWeight: 800,
                fontSize: '1.15rem',
                letterSpacing: '0.06em',
                background: 'linear-gradient(135deg, #FFFFFF 60%, #38BDF8 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              EARTH OBSERVER
            </span>
            <div
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.65rem',
                color: 'var(--cyan)',
                letterSpacing: '0.12em',
                lineHeight: 1,
              }}
            >
              AI SATELLITE INTELLIGENCE
            </div>
          </div>
        </a>

        {/* Desktop Navigation Links */}
        <nav
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '2rem',
          }}
          className="desktop-nav"
        >
          <a href="#hero" className="nav-link">Overview</a>
          <a href="#classify" className="nav-link">Classify Image</a>
          <a href="#insights" className="nav-link">Model Insights</a>
          <a href="#classes" className="nav-link">Classes</a>

          {/* Model Status Pill */}
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.35rem 0.85rem',
              borderRadius: '9999px',
              background: isBackendHealthy
                ? 'rgba(16, 185, 129, 0.1)'
                : 'rgba(244, 63, 94, 0.1)',
              border: `1px solid ${
                isBackendHealthy ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'
              }`,
              fontFamily: 'var(--font-mono)',
              fontSize: '0.75rem',
              color: isBackendHealthy ? '#34D399' : '#FB7185',
            }}
          >
            <span className={`status-dot ${isBackendHealthy ? '' : 'offline'}`} />
            <span>{isBackendHealthy ? `LOCAL MODEL READY (${deviceInfo})` : 'BACKEND OFFLINE'}</span>
          </div>
        </nav>

        {/* Mobile Menu Toggle */}
        <button
          onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
          className="mobile-toggle"
          aria-label="Toggle navigation menu"
          style={{
            background: 'transparent',
            border: 'none',
            color: '#FFFFFF',
            cursor: 'pointer',
            padding: '0.5rem',
            display: 'none',
          }}
        >
          {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile Drawer */}
      {isMobileMenuOpen && (
        <div
          style={{
            background: 'rgba(5, 11, 26, 0.98)',
            borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
            padding: '1.5rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.25rem',
          }}
        >
          <a href="#hero" onClick={() => setIsMobileMenuOpen(false)} className="nav-link">Overview</a>
          <a href="#classify" onClick={() => setIsMobileMenuOpen(false)} className="nav-link">Classify Image</a>
          <a href="#insights" onClick={() => setIsMobileMenuOpen(false)} className="nav-link">Model Insights</a>
          <a href="#classes" onClick={() => setIsMobileMenuOpen(false)} className="nav-link">Classes</a>

          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.5rem 1rem',
              borderRadius: '8px',
              background: isBackendHealthy ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.8rem',
              color: isBackendHealthy ? '#34D399' : '#FB7185',
            }}
          >
            <span className={`status-dot ${isBackendHealthy ? '' : 'offline'}`} />
            <span>{isBackendHealthy ? `LOCAL MODEL READY (${deviceInfo})` : 'BACKEND OFFLINE'}</span>
          </div>
        </div>
      )}

      <style>{`
        .nav-link {
          font-family: var(--font-body);
          font-size: 0.9rem;
          font-weight: 500;
          color: var(--text-secondary);
          transition: color 0.2s ease;
        }
        .nav-link:hover {
          color: var(--cyan);
        }
        @media (max-width: 860px) {
          .desktop-nav {
            display: none !important;
          }
          .mobile-toggle {
            display: block !important;
          }
        }
      `}</style>
    </header>
  );
};
