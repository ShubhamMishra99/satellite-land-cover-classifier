import React, { useState } from 'react';
import { ArrowRight, Cpu, Layers, GitBranch, BarChart3, ShieldCheck, Zap } from 'lucide-react';

export const ModelInsights: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'workflow' | 'benchmark' | 'architectures'>('benchmark');

  const pipelineSteps = [
    { title: 'Satellite Image', subtitle: '64×64 Sentinel-2 Tile', desc: 'RGB 3-channel optical surface reflectance' },
    { title: 'Preprocessing', subtitle: 'Deterministic Pipeline', desc: 'Resize(64,64) → ToTensor() → EuroSAT Normalization' },
    { title: 'Deep Feature Extractor', subtitle: 'CNN / ResNet-18', desc: 'Convolutional hierarchy & residual bottleneck representations' },
    { title: 'Classification Head', subtitle: 'Dense Linear Layer', desc: 'Linear projection to 10 unnormalized class logits' },
    { title: 'Softmax Activation', subtitle: 'Probability Distribution', desc: 'Softmax normalization guaranteeing sum = 1.0' },
    { title: 'Prediction Output', subtitle: 'Top Class & Confidence', desc: 'Final land-use category with Top-3 probability ranking' },
  ];

  const benchmarkData = [
    { metric: 'Test Accuracy', cnn: '75.31%', resnet: '94.99%', delta: '+19.68%', top: 'ResNet-18' },
    { metric: 'Macro F1-Score (Primary)', cnn: '0.7406', resnet: '0.9477', delta: '+0.2071', top: 'ResNet-18' },
    { metric: 'Macro Precision', cnn: '0.7481', resnet: '0.9484', delta: '+0.2003', top: 'ResNet-18' },
    { metric: 'Macro Recall', cnn: '0.7485', resnet: '0.9472', delta: '+0.1987', top: 'ResNet-18' },
    { metric: 'Weighted F1-Score', cnn: '0.7503', resnet: '0.9497', delta: '+0.1994', top: 'ResNet-18' },
    { metric: 'Total Parameters', cnn: '94,762', resnet: '11,181,642', delta: '+11.08M', top: 'Custom CNN (118x lighter)' },
    { metric: 'Inference Latency (Full 4,050)', cnn: '6.71 s', resnet: '10.35 s', delta: '+3.63 s', top: 'Custom CNN' },
    { metric: 'Average Latency per Image', cnn: '1.66 ms', resnet: '2.56 ms', delta: '+0.90 ms', top: 'Custom CNN' },
    { metric: 'Total Test Misclassifications', cnn: '1,000 / 4,050', resnet: '203 / 4,050', delta: '-797 (-79.7%)', top: 'ResNet-18' },
  ];

  return (
    <section
      id="insights"
      style={{
        padding: '5rem 0',
        position: 'relative',
        borderTop: '1px solid rgba(255, 255, 255, 0.06)',
      }}
    >
      <div className="container">
        {/* Section Header */}
        <div style={{ textAlign: 'center', maxWidth: '720px', margin: '0 auto 3rem auto' }}>
          <span className="eyebrow-badge" style={{ marginBottom: '0.85rem' }}>
            <Cpu size={13} />
            ACADEMIC BENCHMARKS & ARCHITECTURE
          </span>
          <h2
            style={{
              fontSize: 'clamp(2rem, 3.5vw, 2.75rem)',
              fontWeight: 800,
              letterSpacing: '-0.02em',
              marginBottom: '0.75rem',
            }}
          >
            Machine Learning Pipeline & Insights
          </h2>
          <p style={{ fontSize: '1.05rem', color: 'var(--text-secondary)' }}>
            Comparing Custom Convolutional Neural Network baseline with ResNet-18 transfer learning on 4,050 untouched test images.
          </p>
        </div>

        {/* Tab Buttons */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            gap: '0.75rem',
            marginBottom: '2.5rem',
          }}
        >
          <button
            onClick={() => setActiveTab('benchmark')}
            className={`btn ${activeTab === 'benchmark' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '0.55rem 1.25rem', fontSize: '0.85rem' }}
          >
            <BarChart3 size={15} />
            <span>Test Benchmark Results (Milestone 5)</span>
          </button>

          <button
            onClick={() => setActiveTab('workflow')}
            className={`btn ${activeTab === 'workflow' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '0.55rem 1.25rem', fontSize: '0.85rem' }}
          >
            <GitBranch size={15} />
            <span>End-to-End Inference Workflow</span>
          </button>

          <button
            onClick={() => setActiveTab('architectures')}
            className={`btn ${activeTab === 'architectures' ? 'btn-primary' : 'btn-secondary'}`}
            style={{ padding: '0.55rem 1.25rem', fontSize: '0.85rem' }}
          >
            <Layers size={15} />
            <span>Model Architectures</span>
          </button>
        </div>

        {/* View 1: Test Benchmark Table */}
        {activeTab === 'benchmark' && (
          <div className="glass-panel" style={{ padding: '2rem', overflowX: 'auto' }}>
            <div style={{ marginBottom: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div>
                <h3 style={{ fontSize: '1.25rem', fontWeight: 700 }}>EuroSAT RGB Final Evaluation (Untouched Test Partition)</h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                  Evaluated across exactly 4,050 test images with zero weight mutations and zero data leakage.
                </p>
              </div>
              <span className="eyebrow-badge" style={{ color: 'var(--cyan)' }}>
                STRICT TEST METRICS
              </span>
            </div>

            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.9rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.12)', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '0.85rem 1rem' }}>Evaluation Metric / Dimension</th>
                  <th style={{ padding: '0.85rem 1rem' }}>Custom CNN (Baseline)</th>
                  <th style={{ padding: '0.85rem 1rem' }}>ResNet-18 (Transfer Learning)</th>
                  <th style={{ padding: '0.85rem 1rem' }}>Delta (ResNet - CNN)</th>
                  <th style={{ padding: '0.85rem 1rem' }}>Top Performer</th>
                </tr>
              </thead>
              <tbody>
                {benchmarkData.map((row, idx) => (
                  <tr
                    key={row.metric}
                    style={{
                      borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                      background: idx % 2 === 0 ? 'rgba(255, 255, 255, 0.015)' : 'transparent',
                    }}
                  >
                    <td style={{ padding: '0.85rem 1rem', fontWeight: 600, color: '#FFFFFF' }}>{row.metric}</td>
                    <td style={{ padding: '0.85rem 1rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{row.cnn}</td>
                    <td style={{ padding: '0.85rem 1rem', color: '#00F0FF', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{row.resnet}</td>
                    <td style={{ padding: '0.85rem 1rem', color: row.delta.startsWith('+') ? '#34D399' : '#38BDF8', fontFamily: 'var(--font-mono)' }}>{row.delta}</td>
                    <td style={{ padding: '0.85rem 1rem' }}>
                      <span
                        style={{
                          display: 'inline-block',
                          padding: '0.2rem 0.6rem',
                          borderRadius: '6px',
                          background: row.top.includes('ResNet') ? 'rgba(0, 240, 255, 0.12)' : 'rgba(129, 140, 248, 0.12)',
                          color: row.top.includes('ResNet') ? 'var(--cyan)' : 'var(--violet)',
                          fontSize: '0.75rem',
                          fontFamily: 'var(--font-mono)',
                        }}
                      >
                        {row.top}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div
              style={{
                marginTop: '1.5rem',
                padding: '0.85rem 1.25rem',
                borderRadius: '8px',
                background: 'rgba(2, 132, 199, 0.08)',
                border: '1px solid rgba(2, 132, 199, 0.2)',
                fontSize: '0.8rem',
                color: 'var(--text-secondary)',
                lineHeight: 1.5,
              }}
            >
              <strong style={{ color: '#FFFFFF' }}>Academic Note on Model Selection:</strong> Macro F1 is our primary model selection metric because it computes unweighted per-class harmonic means across all 10 land-cover classes, preventing performance on majority classes from masking errors on difficult spectral classes like Highway or River.
            </div>
          </div>
        )}

        {/* View 2: End-to-End Workflow Diagram */}
        {activeTab === 'workflow' && (
          <div className="glass-panel" style={{ padding: '2.5rem' }}>
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '2rem', textAlign: 'center' }}>
              Classification Inference Architecture
            </h3>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
                gap: '1.25rem',
                alignItems: 'stretch',
              }}
            >
              {pipelineSteps.map((step, idx) => (
                <div
                  key={step.title}
                  className="glass-card"
                  style={{
                    position: 'relative',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                  }}
                >
                  <div>
                    <div
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '0.7rem',
                        color: 'var(--cyan)',
                        marginBottom: '0.5rem',
                      }}
                    >
                      STEP 0{idx + 1}
                    </div>
                    <h4 style={{ fontSize: '1rem', fontWeight: 700, color: '#FFFFFF', marginBottom: '0.25rem' }}>
                      {step.title}
                    </h4>
                    <div style={{ fontSize: '0.75rem', color: '#38BDF8', marginBottom: '0.65rem' }}>
                      {step.subtitle}
                    </div>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>
                      {step.desc}
                    </p>
                  </div>

                  {idx < pipelineSteps.length - 1 && (
                    <div
                      style={{
                        display: 'none',
                        position: 'absolute',
                        right: '-12px',
                        top: '50%',
                        transform: 'translateY(-50%)',
                        color: 'var(--cyan)',
                        zIndex: 2,
                      }}
                      className="step-arrow"
                    >
                      <ArrowRight size={18} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* View 3: Architecture Cards */}
        {activeTab === 'architectures' && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
              gap: '2rem',
            }}
          >
            {/* Custom CNN Card */}
            <div className="glass-panel" style={{ padding: '2rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1rem' }}>
                <div style={{ padding: '0.5rem', borderRadius: '8px', background: 'rgba(129, 140, 248, 0.15)', color: 'var(--violet)' }}>
                  <Layers size={22} />
                </div>
                <div>
                  <h3 style={{ fontSize: '1.2rem', fontWeight: 700 }}>Custom CNN Baseline</h3>
                  <div style={{ fontSize: '0.75rem', color: 'var(--violet)', fontFamily: 'var(--font-mono)' }}>
                    TRAINED FROM SCRATCH
                  </div>
                </div>
              </div>

              <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
                <li>• <strong>Architecture:</strong> 3-block Convolutional Network with Batch Normalization, ReLU, and Max Pooling.</li>
                <li>• <strong>Feature Learning:</strong> Learns low-to-mid spatial features directly from EuroSAT training tiles.</li>
                <li>• <strong>Parameters:</strong> Lightweight footprint with only 94,762 parameters.</li>
                <li>• <strong>Test Macro F1:</strong> 0.7406 (Test Accuracy: 75.31%).</li>
                <li>• <strong>Role:</strong> Solid reference baseline to measure the true empirical value of transfer learning.</li>
              </ul>
            </div>

            {/* ResNet-18 Card */}
            <div className="glass-panel" style={{ padding: '2rem', borderColor: 'rgba(0, 240, 255, 0.3)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1rem' }}>
                <div style={{ padding: '0.5rem', borderRadius: '8px', background: 'rgba(0, 240, 255, 0.15)', color: 'var(--cyan)' }}>
                  <Cpu size={22} />
                </div>
                <div>
                  <h3 style={{ fontSize: '1.2rem', fontWeight: 700 }}>ResNet-18 Transfer Learning</h3>
                  <div style={{ fontSize: '0.75rem', color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>
                    STAGED FINE-TUNING • RECOMMENDED
                  </div>
                </div>
              </div>

              <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.88rem', color: 'var(--text-secondary)' }}>
                <li>• <strong>Backbone:</strong> ImageNet-pretrained 18-layer Residual Network with skip connections.</li>
                <li>• <strong>Staged Training:</strong> Phase 1 feature extraction (frozen backbone) followed by Phase 2 selective fine-tuning (layer3, layer4, fc).</li>
                <li>• <strong>Parameters:</strong> 11,181,642 total parameters.</li>
                <li>• <strong>Test Macro F1:</strong> 0.9477 (Test Accuracy: 94.99%).</li>
                <li>• <strong>Error Reduction:</strong> Reduces test misclassifications by 79.7% compared to the baseline.</li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};
