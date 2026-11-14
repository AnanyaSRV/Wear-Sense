
import Link from "next/link";

const activities = [
  { number: "01", name: "Walking", type: "DYNAMIC", symbol: "↗", detail: "Continuous locomotion" },
  { number: "02", name: "Walking upstairs", type: "DYNAMIC", symbol: "↑", detail: "Ascending movement" },
  { number: "03", name: "Walking downstairs", type: "DYNAMIC", symbol: "↓", detail: "Descending movement" },
  { number: "04", name: "Sitting", type: "STATIC", symbol: "⌁", detail: "Seated posture" },
  { number: "05", name: "Standing", type: "STATIC", symbol: "│", detail: "Upright posture" },
  { number: "06", name: "Laying", type: "STATIC", symbol: "—", detail: "Horizontal posture" },
];

const researchStats = [
  { value: "06", label: "ACTIVITY CLASSES" },
  { value: "561", label: "ENGINEERED FEATURES" },
  { value: "128", label: "SIGNAL SAMPLES / WINDOW" },
  { value: "95.32%", label: "EXPLORATORY TEST ACCURACY" },
];

function ArrowIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M5 12h14M13 5l7 7-7 7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SignalGraphic() {
  const points = Array.from({ length: 72 }, (_, index) => {
    const x = (index / 71) * 600;
    const wave =
      Math.sin(index * 0.56) * 13 +
      Math.sin(index * 1.27) * 5 +
      Math.cos(index * 0.19) * 8;
    const pulse = Math.exp(-Math.pow((index - 23) / 5, 2)) * 34;
    const secondPulse = Math.exp(-Math.pow((index - 52) / 7, 2)) * 24;
    return `${x},${55 - wave - pulse - secondPulse}`;
  }).join(" ");

  return (
    <svg className="signal-svg" viewBox="0 0 600 110" preserveAspectRatio="none" aria-label="Decorative sensor waveform">
      <defs>
        <linearGradient id="signal-gradient" x1="0" x2="1">
          <stop offset="0%" stopColor="#A78BFA" />
          <stop offset="55%" stopColor="#C4B5FD" />
          <stop offset="100%" stopColor="#C7F36B" />
        </linearGradient>
      </defs>
      <path d="M0 55 H600" className="signal-baseline" />
      <polyline points={points} fill="none" stroke="url(#signal-gradient)" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function MotionFigure() {
  return (
    <div className="motion-figure" aria-label="Abstract illustration of human movement">
      <div className="motion-orbit orbit-one" />
      <div className="motion-orbit orbit-two" />
      <div className="motion-orbit orbit-three" />
      <div className="motion-crosshair crosshair-one" />
      <div className="motion-crosshair crosshair-two" />
      <div className="motion-person">
        <div className="person-head" />
        <div className="person-neck" />
        <div className="person-torso" />
        <div className="person-arm arm-left" />
        <div className="person-arm arm-right" />
        <div className="person-leg leg-left" />
        <div className="person-leg leg-right" />
        <div className="person-joint joint-shoulder-left" />
        <div className="person-joint joint-shoulder-right" />
        <div className="person-joint joint-hip" />
        <div className="person-joint joint-knee-left" />
        <div className="person-joint joint-knee-right" />
      </div>
      <span className="motion-coordinate coordinate-top">X 04.82 / Y 12.06</span>
      <span className="motion-coordinate coordinate-bottom">MOTION FIELD 01</span>
      <span className="motion-tag tag-one">ACCELERATION</span>
      <span className="motion-tag tag-two">GYROSCOPE</span>
      <div className="motion-center-dot" />
    </div>
  );
}

export default function Home() {
  return (
    <main>
      <div className="announcement-bar">
        <span className="announcement-pulse" />
        <span>EXPLORING THE INTERSECTION OF AI, MOVEMENT & SMART TEXTILES</span>
        <span className="announcement-index">RESEARCH PROJECT / 001</span>
      </div>

      <header className="site-header">
        <Link href="/" className="brand" aria-label="WearSense home">
          <span className="brand-symbol"><span /><span /><span /></span>
          <span>wear<span className="brand-light">sense</span><sup>®</sup></span>
        </Link>

        <nav className="desktop-nav" aria-label="Main navigation">
          <a href="#about">ABOUT</a>
          <a href="#activities">MOVEMENT</a>
          <a href="#technology">TECHNOLOGY</a>
          <a href="#research">RESEARCH</a>
        </nav>

        <a href="#explore" className="header-cta">
          EXPLORE THE SYSTEM <ArrowIcon />
        </a>
      </header>

      <section className="hero" id="explore">
        <div className="hero-grid" />
        <div className="hero-main">
          <div className="hero-copy">
            <div className="eyebrow hero-eyebrow">
              <span className="eyebrow-line" />
              INTELLIGENCE IN EVERY MOVEMENT
            </div>

            <h1>
              The future
              <br />
              <span className="hero-outline">of movement</span>
              <br />
              is <span className="hero-highlight">intelligent.</span>
            </h1>

            <p className="hero-description">
              Understanding human activity through artificial intelligence,
              wearable sensing and the language of movement.
            </p>

            <div className="hero-actions">
              <a href="#activities" className="button-primary">
                EXPLORE MOVEMENT <ArrowIcon />
              </a>
              <a href="#about" className="button-text">
                DISCOVER WEARSENSE <span>↘</span>
              </a>
            </div>

            <div className="hero-footnote">
              <span className="footnote-number">01 / 06</span>
              <span>HUMAN ACTIVITY RECOGNITION</span>
              <span className="footnote-rule" />
            </div>
          </div>

          <div className="hero-visual">
            <div className="visual-topline">
              <span>FIG. 001 — MOVEMENT STUDY</span>
              <span className="live-indicator"><i /> SENSOR CONCEPT</span>
            </div>

            <MotionFigure />

            <div className="visual-bottomline">
              <div>
                <span className="micro-label">MOVEMENT INTELLIGENCE</span>
                <strong>BEYOND THE SURFACE.</strong>
              </div>
              <span className="visual-coordinate">17° 23&apos; 12.4&quot; N</span>
            </div>
          </div>
        </div>

        <div className="hero-bottom">
          <span>DESIGNED TO UNDERSTAND THE HUMAN BODY.</span>
          <div className="hero-bottom-center">
            <span className="scroll-line" />
            SCROLL TO EXPLORE
          </div>
          <span>AI / SENSING / ACTIVEWEAR</span>
        </div>
      </section>

      <section className="intro-section section-padding" id="about">
        <div className="section-kicker">
          <span>01</span>
          <span className="kicker-line" />
          <span>THE IDEA</span>
        </div>

        <div className="intro-content">
          <div className="intro-heading">
            <p className="small-overline">A NEW PERSPECTIVE ON ACTIVITY.</p>
            <h2>
              Movement is data.
              <br />
              <span>We give it meaning.</span>
            </h2>
          </div>

          <div className="intro-text">
            <p>
              WearSense is an AI-powered human activity recognition research
              prototype exploring how wearable sensor data can be translated
              into meaningful insights about everyday movement and posture.
            </p>
            <p>
              By combining inertial sensor features with machine learning,
              the project investigates a foundation for more intelligent,
              responsive activewear experiences.
            </p>
            <a href="#technology" className="inline-link">
              EXPLORE THE TECHNOLOGY <ArrowIcon />
            </a>
          </div>
        </div>

        <div className="stats-grid">
          {researchStats.map((stat) => (
            <div className="stat-item" key={stat.label}>
              <strong>{stat.value}</strong>
              <span>{stat.label}</span>
            </div>
          ))}
        </div>
        <p className="stats-note">
          * Test results are exploratory: the supplied test partition was
          inspected during development and is not an untouched independent evaluation.
        </p>
      </section>

      <section className="activities-section section-padding" id="activities">
        <div className="section-kicker light-kicker">
          <span>02</span>
          <span className="kicker-line" />
          <span>THE MOVEMENT INDEX</span>
        </div>

        <div className="activities-heading">
          <div>
            <p className="small-overline">SIX WAYS TO MOVE.</p>
            <h2>
              Every movement
              <br />
              <span>has a signature.</span>
            </h2>
          </div>
          <p>
            Six everyday activities form the foundation of our human activity
            recognition experiment. Explore the movement categories behind
            the model.
          </p>
        </div>

        <div className="activity-grid">
          {activities.map((activity) => (
            <article className="activity-card" key={activity.number}>
              <div className="activity-card-top">
                <span>{activity.number} / 06</span>
                <span className="activity-type">{activity.type}</span>
              </div>
              <div className="activity-symbol">{activity.symbol}</div>
              <div className="activity-card-bottom">
                <div>
                  <h3>{activity.name}</h3>
                  <p>{activity.detail}</p>
                </div>
                <span className="activity-arrow">↗</span>
              </div>
            </article>
          ))}
        </div>

        <div className="activity-footer">
          <span>UCI HUMAN ACTIVITY RECOGNITION DATASET</span>
          <span>6 CLASSES / 1 RESEARCH FRAMEWORK</span>
        </div>
      </section>

      <section className="sensor-section section-padding" id="technology">
        <div className="section-kicker">
          <span>03</span>
          <span className="kicker-line" />
          <span>INSIDE THE SIGNAL</span>
        </div>

        <div className="sensor-layout">
          <div className="sensor-copy">
            <p className="small-overline">FROM SIGNAL TO UNDERSTANDING.</p>
            <h2>
              The body
              <br />
              speaks in
              <span className="violet-text"> signals.</span>
            </h2>
            <p className="sensor-description">
              Accelerometers and gyroscopes capture movement patterns.
              Machine learning interprets sensor-derived features to
              distinguish activities and postures.
            </p>

            <div className="sensor-list">
              <div><span>01</span> INERTIAL SENSING</div>
              <div><span>02</span> SIGNAL PROCESSING</div>
              <div><span>03</span> MACHINE LEARNING</div>
              <div><span>04</span> ACTIVITY RECOGNITION</div>
            </div>
          </div>

          <div className="sensor-visual">
            <div className="sensor-visual-header">
              <span>INERTIAL SIGNAL / CONCEPT PREVIEW</span>
              <span className="sensor-dot">● DATA EXPLORATION</span>
            </div>
            <div className="signal-label-row">
              <span>BODY ACCELERATION · X AXIS</span>
              <span>128 POINTS / WINDOW</span>
            </div>
            <SignalGraphic />
            <div className="signal-axis">
              <span>00</span><span>32</span><span>64</span><span>96</span><span>128</span>
            </div>
            <div className="signal-divider" />
            <div className="signal-legend">
              <span><i className="legend-violet" /> ACCELERATION</span>
              <span><i className="legend-lime" /> MOVEMENT PATTERN</span>
            </div>
            <p className="signal-disclaimer">
              Decorative waveform illustration. The interactive Sensor Lab
              will display actual UCI HAR recordings.
            </p>
          </div>
        </div>
      </section>

      <section className="ai-section section-padding" id="research">
        <div className="section-kicker light-kicker">
          <span>04</span>
          <span className="kicker-line" />
          <span>THE INTELLIGENCE LAYER</span>
        </div>

        <div className="ai-heading">
          <p className="small-overline">LEARNING FROM MOVEMENT.</p>
          <h2>
            Not just sensing.
            <br />
            <span>Understanding.</span>
          </h2>
          <p>
            A grouped-validation-selected RBF Support Vector Machine
            translates 561 sensor-derived features into six activity classes.
          </p>
        </div>

        <div className="ai-panel">
          <div className="ai-panel-top">
            <span>MODEL PROFILE / 001</span>
            <span className="model-status"><i /> SAVED MODEL VERIFIED</span>
          </div>
          <div className="ai-panel-main">
            <div className="ai-model-name">
              <span className="micro-label">SELECTED CLASSIFIER</span>
              <strong>RBF <span>SVM</span></strong>
              <p>Radial Basis Function · Support Vector Machine</p>
            </div>
            <div className="ai-metric">
              <span className="micro-label">EXPLORATORY TEST ACCURACY</span>
              <strong>95.32<span>%</span></strong>
              <div className="metric-track"><i /></div>
            </div>
          </div>
          <div className="ai-panel-bottom">
            <span>C = 3.0</span>
            <span>GAMMA = 0.001</span>
            <span>STANDARD SCALER</span>
            <span>2,947 TEST SAMPLES</span>
          </div>
        </div>
        <p className="ai-caveat">
          Results are exploratory and do not represent a final independent
          benchmark. Model probabilities are not independently calibrated.
        </p>
      </section>

      <section className="closing-section section-padding">
        <div className="closing-decoration">W/S</div>
        <p className="small-overline">THE NEXT LAYER OF WEARABLE INTELLIGENCE.</p>
        <h2>
          Rethink what
          <br />
          <span>activewear can do.</span>
        </h2>
        <p className="closing-description">
          From movement recognition to intelligent textiles, WearSense
          explores a more connected relationship between people, technology
          and the clothes they move in.
        </p>
        <a href="#about" className="button-primary closing-button">
          EXPLORE THE PROJECT <ArrowIcon />
        </a>
      </section>

      <footer className="site-footer">
        <Link href="/" className="brand footer-brand">
          <span className="brand-symbol"><span /><span /><span /></span>
          <span>wear<span className="brand-light">sense</span><sup>®</sup></span>
        </Link>
        <span>AI-POWERED HUMAN ACTIVITY RECOGNITION</span>
        <span>RESEARCH PROTOTYPE · 2026</span>
        <a href="#explore" className="back-to-top">BACK TO TOP ↑</a>
      </footer>
    </main>
  );
}
