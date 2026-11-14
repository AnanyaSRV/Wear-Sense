
"use client";

import { useCallback, useEffect, useState } from "react";
import styles from "./dashboard.module.css";

const API_URL =
  process.env.NEXT_PUBLIC_WEARSENSE_API_URL || "http://127.0.0.1:8000";

type Probability = {
  activity_id: number;
  activity: string;
  probability: number;
};

type PredictionResponse = {
  sample_index: number;
  subject_id: number;
  actual_activity_id: number;
  actual_activity: string;
  prediction: {
    predicted_activity_id: number;
    predicted_activity: string;
    probabilities: Probability[];
    probability_note?: string;
  };
  data_source: string;
  is_live_sensor_data: boolean;
};

type HealthResponse = {
  status: string;
  model_loaded: boolean;
  dataset_loaded: boolean;
};

const activities = [
  {
    id: 1,
    name: "Walking",
    type: "Dynamic movement",
    description: "Regular walking motion.",
  },
  {
    id: 2,
    name: "Walking Upstairs",
    type: "Dynamic movement",
    description: "Movement while ascending stairs.",
  },
  {
    id: 3,
    name: "Walking Downstairs",
    type: "Dynamic movement",
    description: "Movement while descending stairs.",
  },
  {
    id: 4,
    name: "Sitting",
    type: "Static posture",
    description: "A seated body posture.",
  },
  {
    id: 5,
    name: "Standing",
    type: "Static posture",
    description: "An upright stationary posture.",
  },
  {
    id: 6,
    name: "Laying",
    type: "Static posture",
    description: "A horizontal resting posture.",
  },
];

function Brand() {
  return (
    <div className={styles.brand}>
      <span className={styles.brandMark}>
        <i />
        <i />
        <i />
      </span>
      <span className={styles.brandName}>
        wear<span>sense</span>
        <sup>®</sup>
      </span>
    </div>
  );
}

function formatPercentage(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState("");

  const [sampleIndex, setSampleIndex] = useState(0);
  const [sampleInput, setSampleInput] = useState("0");

  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [expandedActivity, setExpandedActivity] = useState<number | null>(1);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activeNav, setActiveNav] = useState("Overview");

  const checkBackend = useCallback(async () => {
    try {
      setHealthError("");

      const response = await fetch(`${API_URL}/health`, {
        method: "GET",
        cache: "no-store",
      });

      if (!response.ok) {
        throw new Error(`Health check failed (${response.status})`);
      }

      const data: HealthResponse = await response.json();
      setHealth(data);
    } catch (err) {
      setHealth(null);
      setHealthError(
        err instanceof Error
          ? err.message
          : "Could not connect to the backend."
      );
    }
  }, []);

  const loadSample = useCallback(async (index: number) => {
    if (!Number.isInteger(index) || index < 0) {
      setError("Enter a valid non-negative sample index.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(
        `${API_URL}/predict/sample/${index}`,
        {
          method: "POST",
          headers: {
            Accept: "application/json",
          },
          cache: "no-store",
        }
      );

      if (!response.ok) {
        let message = `Prediction request failed (${response.status}).`;

        try {
          const errorData = await response.json();
          if (errorData.detail) {
            message = String(errorData.detail);
          }
        } catch {
          // Keep the HTTP error message if the response is not JSON.
        }

        throw new Error(message);
      }

      const data: PredictionResponse = await response.json();

      if (
        typeof data.sample_index !== "number" ||
        typeof data.actual_activity !== "string" ||
        !data.prediction ||
        typeof data.prediction.predicted_activity !== "string" ||
        !Array.isArray(data.prediction.probabilities)
      ) {
        throw new Error("The backend returned an unexpected prediction response.");
      }

      setResult(data);
      setSampleIndex(index);
      setSampleInput(String(index));
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to load the selected sample."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkBackend();
    loadSample(0);
  }, [checkBackend, loadSample]);

  const predictionCorrect =
    result !== null &&
    result.prediction.predicted_activity_id === result.actual_activity_id;

  const topProbability =
    result?.prediction.probabilities.reduce(
      (highest, item) =>
        item.probability > highest.probability ? item : highest,
      result.prediction.probabilities[0]
    );

  const navigateTo = (label: string, id: string) => {
    setActiveNav(label);
    setMobileMenuOpen(false);
    document.getElementById(id)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  };

  const previousSample = () => {
    if (sampleIndex > 0) {
      loadSample(sampleIndex - 1);
    }
  };

  const nextSample = () => {
    loadSample(sampleIndex + 1);
  };

  return (
    <main className={styles.dashboard}>
      <aside
        className={`${styles.sidebar} ${
          mobileMenuOpen ? styles.sidebarOpen : ""
        }`}
      >
        <div className={styles.sidebarTop}>
          <a href="/" aria-label="WearSense home">
            <Brand />
          </a>

          <button
            className={styles.closeMenu}
            onClick={() => setMobileMenuOpen(false)}
            aria-label="Close navigation"
          >
            ×
          </button>
        </div>

        <div className={styles.workspaceLabel}>WORKSPACE</div>

        <div className={styles.workspaceCard}>
          <div className={styles.workspaceAvatar}>W</div>
          <div>
            <strong>WearSense Lab</strong>
            <span>Research workspace</span>
          </div>
        </div>

        <div className={styles.navLabel}>MONITOR</div>

        <nav className={styles.navigation}>
          {[
            ["Overview", "overview"],
            ["Prediction", "prediction"],
            ["Activities", "activities"],
            ["Dataset", "dataset"],
          ].map(([label, id]) => (
            <button
              key={id}
              className={`${styles.navItem} ${
                activeNav === label ? styles.navActive : ""
              }`}
              onClick={() => navigateTo(label, id)}
            >
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className={styles.sidebarBottom}>
          <div className={styles.sidebarOrb} />
          <span className={styles.sidebarOverline}>WEARSENSE / 2.0</span>
          <strong>
            Movement,
            <br />
            decoded.
          </strong>
          <p>Human activity recognition using machine learning.</p>
          <div className={styles.sidebarVersion}>
            BIO-DIGITAL INTERFACE <span>V2.0</span>
          </div>
        </div>
      </aside>

      {mobileMenuOpen && (
        <button
          className={styles.mobileBackdrop}
          onClick={() => setMobileMenuOpen(false)}
          aria-label="Close navigation"
        />
      )}

      <section className={styles.mainPanel}>
        <header className={styles.topbar}>
          <button
            className={styles.mobileMenuButton}
            onClick={() => setMobileMenuOpen(true)}
            aria-label="Open navigation"
          >
            ☰
          </button>

          <div className={styles.breadcrumb}>
            WEARSENSE <span>/</span> <strong>{activeNav.toUpperCase()}</strong>
          </div>

          <div className={styles.topbarActions}>
            <div className={styles.demoBadge}>
              <span />
              DATASET MODE
            </div>

            <div className={styles.profile}>
              <div className={styles.profileAvatar}>W</div>
              <div>
                <strong>WearSense</strong>
                <span>Research prototype</span>
              </div>
            </div>
          </div>
        </header>

        <div className={styles.content}>
          <section className={styles.welcomeSection} id="overview">
            <div>
              <div className={styles.eyebrow}>
                <span />
                HUMAN ACTIVITY RECOGNITION / MACHINE LEARNING
              </div>

              <h1>
                Movement <span>intelligence.</span>
              </h1>

              <p className={styles.welcomeText}>
                Explore real model predictions from the UCI Human Activity
                Recognition dataset.
              </p>
            </div>

            <div className={styles.welcomeActions}>
              <button
                className={styles.secondaryButton}
                onClick={() => {
                  checkBackend();
                  loadSample(sampleIndex);
                }}
                disabled={loading}
              >
                ↻ Refresh data
              </button>
            </div>
          </section>

          <section className={styles.statusStrip}>
            <div className={styles.statusLeft}>
              <span className={styles.statusPulse} />
              <strong>
                {health?.status === "healthy"
                  ? "BACKEND CONNECTED"
                  : "BACKEND UNAVAILABLE"}
              </strong>
              <span className={styles.statusDivider}>/</span>
              <span>
                {health?.model_loaded && health?.dataset_loaded
                  ? "Model and dataset loaded"
                  : "Checking model and dataset"}
              </span>
            </div>

            <div className={styles.statusRight}>
              UCI HAR <span>·</span> TEST DATASET
            </div>
          </section>

          {healthError && (
            <div className={styles.panel} style={{ padding: 16, marginBottom: 16 }}>
              <strong>Backend connection error</strong>
              <p>{healthError}</p>
              <p>
                Check that your existing FastAPI server is running at{" "}
                <code>{API_URL}</code>.
              </p>
            </div>
          )}

          <section
            className={styles.featureGrid}
            id="prediction"
          >
            <article className={`${styles.panel} ${styles.movementPanel}`}>
              <div className={styles.panelHeader}>
                <div>
                  <span className={styles.panelEyebrow}>
                    MODEL INFERENCE
                  </span>
                  <h2>Activity prediction</h2>
                </div>

                <span className={styles.demoTag}>REAL MODEL OUTPUT</span>
              </div>

              <div style={{ marginTop: 25 }}>
                <span className={styles.panelEyebrow}>
                  PREDICTED ACTIVITY
                </span>

                <h2
                  style={{
                    fontSize: 30,
                    color: "#d9ff9c",
                    marginTop: 10,
                  }}
                >
                  {loading
                    ? "Processing..."
                    : result?.prediction.predicted_activity || "Waiting for data"}
                </h2>

                <p className={styles.sectionIntro}>
                  The activity class returned by the trained RBF SVM model.
                </p>
              </div>

              <div className={styles.movementFooter}>
                <div>
                  <span>TEST SAMPLE</span>
                  <strong>{result?.sample_index ?? "—"}</strong>
                </div>

                <div>
                  <span>SUBJECT ID</span>
                  <strong>{result?.subject_id ?? "—"}</strong>
                </div>

                <div>
                  <span>TOP CLASS PROBABILITY</span>
                  <strong>
                    {topProbability
                      ? formatPercentage(topProbability.probability)
                      : "—"}
                  </strong>
                </div>
              </div>
            </article>

            <article className={`${styles.panel} ${styles.goalPanel}`}>
              <div className={styles.panelHeader}>
                <div>
                  <span className={styles.panelEyebrow}>
                    GROUND TRUTH COMPARISON
                  </span>
                  <h2>Prediction result</h2>
                </div>
              </div>

              {result ? (
                <>
                  <div className={styles.goalVisual}>
                    <div className={styles.goalRing}>
                      <strong>{predictionCorrect ? "✓" : "!"}</strong>
                      <span>
                        {predictionCorrect ? "MATCH" : "MISMATCH"}
                      </span>
                    </div>
                  </div>

                  <div className={styles.goalNumber}>
                    {predictionCorrect ? "Correct" : "Incorrect"}
                  </div>

                  <p className={styles.goalDescription}>
                    {predictionCorrect
                      ? "The predicted activity matches the actual dataset label."
                      : "The predicted activity differs from the actual dataset label."}
                  </p>
                </>
              ) : (
                <p className={styles.goalDescription}>
                  Waiting for a successful prediction.
                </p>
              )}
            </article>
          </section>

          <section className={`${styles.panel} ${styles.chartPanel}`}>
            <div className={styles.panelHeader}>
              <div>
                <span className={styles.panelEyebrow}>
                  TEST DATA EXPLORER
                </span>
                <h2>Select a dataset sample</h2>
              </div>

              <span className={styles.demoTag}>UCI HAR</span>
            </div>

            <p className={styles.sectionIntro}>
              Select a sample index from the test dataset. The existing
              backend will run the trained model on that sample and return
              its prediction.
            </p>

            <form
              onSubmit={(event) => {
                event.preventDefault();
                const index = Number(sampleInput);
                loadSample(index);
              }}
              style={{
                display: "flex",
                flexWrap: "wrap",
                alignItems: "center",
                gap: 10,
                marginTop: 20,
              }}
            >
              <input
                type="number"
                min="0"
                step="1"
                value={sampleInput}
                onChange={(event) => setSampleInput(event.target.value)}
                aria-label="Test dataset sample index"
                style={{
                  width: 150,
                  padding: "12px 14px",
                  color: "#edf5e9",
                  background: "#101b16",
                  border: "1px solid rgba(188,224,177,.25)",
                  borderRadius: 7,
                  outline: "none",
                }}
              />

              <button
                type="submit"
                className={styles.liveButton}
                disabled={loading}
              >
                {loading ? "Loading..." : "Run prediction"}
              </button>

              <button
                type="button"
                className={styles.secondaryButton}
                onClick={previousSample}
                disabled={loading || sampleIndex === 0}
              >
                Previous
              </button>

              <button
                type="button"
                className={styles.secondaryButton}
                onClick={nextSample}
                disabled={loading}
              >
                Next
              </button>
            </form>

            {error && (
              <p
                role="alert"
                style={{
                  color: "#ff9d91",
                  marginTop: 15,
                  fontSize: 12,
                }}
              >
                {error}
              </p>
            )}

            {result && (
              <div style={{ marginTop: 22 }}>
                <div className={styles.movementFooter}>
                  <div>
                    <span>ACTUAL ACTIVITY</span>
                    <strong>{result.actual_activity}</strong>
                  </div>

                  <div>
                    <span>PREDICTED ACTIVITY</span>
                    <strong>{result.prediction.predicted_activity}</strong>
                  </div>

                  <div>
                    <span>RESULT</span>
                    <strong>{predictionCorrect ? "MATCH" : "MISMATCH"}</strong>
                  </div>
                </div>
              </div>
            )}
          </section>

          <section className={`${styles.panel} ${styles.insightPanel}`}>
            <div className={styles.panelHeader}>
              <div>
                <span className={styles.panelEyebrow}>
                  CLASS DISTRIBUTION
                </span>
                <h2>Prediction probabilities</h2>
              </div>

              <span className={styles.demoTag}>MODEL OUTPUT</span>
            </div>

            {result ? (
              <>
                <p className={styles.sectionIntro}>
                  Probability values returned by the model for the selected
                  test sample.
                </p>

                <div style={{ display: "grid", gap: 17, marginTop: 20 }}>
                  {result.prediction.probabilities
                    .slice()
                    .sort((a, b) => b.probability - a.probability)
                    .map((item) => {
                      const isPredicted =
                        item.activity_id ===
                        result.prediction.predicted_activity_id;

                      return (
                        <div key={item.activity_id}>
                          <div
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              gap: 10,
                              marginBottom: 7,
                              fontSize: 10,
                            }}
                          >
                            <strong
                              style={{
                                color: isPredicted ? "#d9ff9c" : "#c1d1b8",
                              }}
                            >
                              {item.activity}
                              {isPredicted ? " · PREDICTED" : ""}
                            </strong>

                            <span style={{ color: "#a5b99d" }}>
                              {formatPercentage(item.probability)}
                            </span>
                          </div>

                          <div
                            style={{
                              width: "100%",
                              height: 7,
                              borderRadius: 20,
                              background: "rgba(172,210,151,.13)",
                              overflow: "hidden",
                            }}
                          >
                            <div
                              style={{
                                width: `${Math.max(
                                  0,
                                  Math.min(100, item.probability * 100)
                                )}%`,
                                height: "100%",
                                borderRadius: 20,
                                background: isPredicted
                                  ? "linear-gradient(90deg,#8dcf83,#d9ff9c)"
                                  : "linear-gradient(90deg,#4c987d,#76dfc0)",
                                transition: "width .35s ease",
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                </div>

                <p className={styles.sectionIntro} style={{ marginTop: 20 }}>
                  {result.prediction.probability_note ||
                    "These are model probability outputs, not calibrated certainty estimates."}
                </p>
              </>
            ) : (
              <p className={styles.sectionIntro}>
                Probability distribution will appear after a successful
                prediction.
              </p>
            )}
          </section>

          <section
            className={`${styles.panel} ${styles.expandSection}`}
            id="activities"
          >
            <div className={styles.panelHeader}>
              <div>
                <span className={styles.panelEyebrow}>
                  RECOGNIZED ACTIVITY CLASSES
                </span>
                <h2>Activity library</h2>
              </div>

              <span className={styles.demoTag}>06 CLASSES</span>
            </div>

            <p className={styles.sectionIntro}>
              The six activity classes available in the UCI HAR dataset.
              Select a class to explore its description.
            </p>

            <div className={styles.activityCards}>
              {activities.map((activity) => {
                const expanded = expandedActivity === activity.id;

                return (
                  <button
                    type="button"
                    key={activity.id}
                    className={`${styles.activityCard} ${
                      expanded ? styles.activityCardExpanded : ""
                    }`}
                    onClick={() =>
                      setExpandedActivity(expanded ? null : activity.id)
                    }
                    aria-expanded={expanded}
                  >
                    <span className={styles.activityCardTop}>
                      <span className={styles.activityCardNumber}>
                        0{activity.id}
                      </span>
                      <span className={styles.activityCardType}>
                        {activity.type}
                      </span>
                    </span>

                    <span className={styles.activityCardSymbol}>
                      {["↗", "↑", "↓", "▤", "│", "━"][activity.id - 1]}
                    </span>

                    <span className={styles.activityCardName}>
                      {activity.name}
                    </span>

                    <span className={styles.activityCardDescription}>
                      {activity.description}
                    </span>

                    <span className={styles.activityCardBottom}>
                      {expanded ? "SELECTED CLASS" : "EXPLORE CLASS"}
                      <span className={styles.expandGlyph}>
                        {expanded ? "−" : "+"}
                      </span>
                    </span>

                    {expanded && (
                      <span className={styles.activityCardDetail}>
                        {activity.description}
                        <span className={styles.detailTag}>
                          UCI HAR · ACTIVITY ID {activity.id}
                        </span>
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </section>

          <section
            className={`${styles.panel} ${styles.sensorPanel}`}
            id="dataset"
          >
            <div className={styles.panelHeader}>
              <div>
                <span className={styles.panelEyebrow}>
                  DATASET INFORMATION
                </span>
                <h2>UCI Human Activity Recognition</h2>
              </div>

              <span className={styles.sensorStatus}>
                <i />
                DATASET SOURCE
              </span>
            </div>

            <p className={styles.sectionIntro}>
              WearSense uses the UCI Human Activity Recognition Using
              Smartphones dataset. The selected model receives preprocessed
              sensor-derived feature vectors rather than live sensor readings.
            </p>

            <div className={styles.sensorCards}>
              <article className={styles.sensorCard}>
                <div className={styles.sensorInfo}>
                  <strong>Accelerometer</strong>
                  <small>Smartphone inertial-sensor data</small>
                </div>
              </article>

              <article className={styles.sensorCard}>
                <div className={styles.sensorInfo}>
                  <strong>Gyroscope</strong>
                  <small>Smartphone rotational-sensor data</small>
                </div>
              </article>

              <article className={styles.sensorCard}>
                <div className={styles.sensorInfo}>
                  <strong>561 features</strong>
                  <small>Preprocessed feature vector used by the model</small>
                </div>
              </article>
            </div>

            <div className={styles.sensorFoot}>
              <span>DATA SOURCE</span>
              <strong>UCI HAR TEST PARTITION</strong>
            </div>

            <div className={styles.sensorFoot}>
              <span>LIVE WEARABLE INPUT</span>
              <strong>NOT CONNECTED</strong>
            </div>

            <p className={styles.sectionIntro}>
              Dataset reference:{" "}
              <a
                href="https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones"
                target="_blank"
                rel="noreferrer"
                style={{ color: "#c5ed91" }}
              >
                UCI Machine Learning Repository
              </a>
            </p>
          </section>

          <footer className={styles.footer}>
            <span>
              WEARSENSE 2.0 <i /> HUMAN ACTIVITY RECOGNITION
            </span>

            <span>
              RESEARCH PROTOTYPE · NOT A MEDICAL DEVICE
            </span>
          </footer>
        </div>
      </section>
    </main>
  );
}
