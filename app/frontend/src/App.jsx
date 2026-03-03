import { useState, useEffect, useRef } from "react";

const FONTS_CSS = `
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;1,8..60,300&display=swap');
`;

export default function SemanticSearch() {
  const [query, setQuery] = useState("");
  const [activeQuery, setActiveQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    const style = document.createElement("style");
    style.textContent = FONTS_CSS;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

  const handleSearch = async (overrideQuery) => {
  const q = overrideQuery || query;
  if (!q.trim()) return;

  setLoading(true);
  setSearched(true);
  setActiveQuery(q);

  try {
    const res = await fetch(`${API_URL}/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q.trim(), top_k: 8 }),
    });

    const data = await res.json();
    setResults(data.results || []);
  } catch (err) {
    console.error(err);
    setResults([]);
  } finally {
    setLoading(false);
  }
};

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleSearch();
  };

  const formatYear = (date) => (date ? date.split("-")[0] : "");
  const scoreToPercent = (score) => Math.round(score * 100);

  const handleReset = () => {
    setSearched(false);
    setResults([]);
    setQuery("");
    setActiveQuery("");
    setTimeout(() => inputRef.current?.focus(), 300);
  };

  return (
    <div style={styles.root}>
      {/* Brighter background with visible hue */}
      <div style={styles.bgBase} />
      <div style={styles.bgOrb1} />
      <div style={styles.bgOrb2} />
      <div style={styles.bgOrb3} />
      <div style={styles.bgOrb4} />
      <div style={styles.bgHaze} />
      <div style={styles.noiseOverlay} />

      <div style={styles.page}>
        {/* ── HERO / HEADER ─────────────────────── */}
        <div style={{
          ...styles.hero,
          ...(searched ? styles.heroCompact : {}),
        }}>
          {/* Logo — click to reset */}
          <div onClick={handleReset} style={{
            ...styles.logo,
            ...(searched ? styles.logoSmall : {}),
          }}>
            <svg width="40" height="40" viewBox="0 0 40 40" fill="none"
              style={{ transition: "all 0.5s ease", width: searched ? 28 : 40, height: searched ? 28 : 40 }}>
              <circle cx="20" cy="20" r="17" stroke="url(#g1)" strokeWidth="1.5" />
              <circle cx="20" cy="20" r="9" stroke="url(#g1)" strokeWidth="1" opacity="0.5" />
              <circle cx="20" cy="20" r="3" fill="url(#g1)" />
              <defs>
                <linearGradient id="g1" x1="0" y1="0" x2="40" y2="40">
                  <stop offset="0%" stopColor="#e0cbff" />
                  <stop offset="100%" stopColor="#6ee7b7" />
                </linearGradient>
              </defs>
            </svg>
          </div>

          {/* Title + subtitle — collapse on search */}
          <div style={{
            ...styles.titleWrap,
            ...(searched ? styles.titleWrapHidden : {}),
          }}>
            <h1 style={styles.title}>nebula</h1>
            <p style={styles.subtitle}>
              describe a mood, a feeling, a vibe and we'll find the film
            </p>
          </div>

          {/* Search bar */}
          <div style={{
            ...styles.searchWrap,
            ...(searched ? styles.searchWrapCompact : {}),
          }}>
            <div style={styles.searchGlow} />
            <div style={styles.searchBar}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
                stroke="rgba(224,203,255,0.45)" strokeWidth="2" strokeLinecap="round"
                style={{ flexShrink: 0 }}>
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="a movie that feels like a late night drive..."
                style={styles.input}
                spellCheck={false}
              />
              {query && (
                <button onClick={() => { setQuery(""); inputRef.current?.focus(); }}
                  style={styles.clearBtn}>×</button>
              )}
              <button onClick={() => handleSearch()}
                style={{ ...styles.goBtn, opacity: query.trim() ? 1 : 0.25 }}
                disabled={!query.trim() || loading}>
                {loading
                  ? <div style={styles.spinSm} />
                  : <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
                      stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                      <line x1="5" y1="12" x2="19" y2="12" />
                      <polyline points="12 5 19 12 12 19" />
                    </svg>
                }
              </button>
            </div>
          </div>

          {/* Chips — collapse on search */}
          <div style={{
            ...styles.chips,
            ...(searched ? styles.chipsHidden : {}),
          }}>
            {["something cozy for a rainy day", "mind-bending sci-fi",
              "movies like Blade Runner", "a heist gone wrong"].map((s) => (
              <button key={s} style={styles.chip}
                onClick={() => { setQuery(s); handleSearch(s); }}
                onMouseEnter={(e) => {
                  e.target.style.background = "rgba(224,203,255,0.14)";
                  e.target.style.borderColor = "rgba(224,203,255,0.35)";
                  e.target.style.color = "rgba(224,203,255,0.75)";
                }}
                onMouseLeave={(e) => {
                  e.target.style.background = "rgba(224,203,255,0.06)";
                  e.target.style.borderColor = "rgba(224,203,255,0.15)";
                  e.target.style.color = "rgba(224,203,255,0.5)";
                }}>
                {s}
              </button>
            ))}
          </div>
        </div>

        {/* ── RESULTS ───────────────────────────── */}
        {searched && (
          <div style={styles.resultsWrap}>
            {loading ? (
              <div style={styles.loadWrap}>
                <div style={styles.spinLg} />
                <p style={styles.loadText}>searching...</p>
              </div>
            ) : results.length === 0 ? (
              <p style={styles.empty}>no matches found</p>
            ) : (
              <>
                <p style={styles.resultsLabel}>
                  results for <span style={styles.labelQ}>"{activeQuery}"</span>
                </p>

                <div style={styles.list}>
                  {results.map((m, i) => (
                    <div key={m.tmdb_id}
                      style={{ ...styles.card, animationDelay: `${i * 65}ms` }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = "rgba(28,22,50,0.8)";
                        e.currentTarget.style.borderColor = "rgba(224,203,255,0.22)";
                        e.currentTarget.style.transform = "translateX(4px)";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = "rgba(22,16,42,0.55)";
                        e.currentTarget.style.borderColor = "rgba(224,203,255,0.08)";
                        e.currentTarget.style.transform = "translateX(0)";
                      }}>
                      {/* Rank */}
                      <div style={styles.rank}>
                        <span style={styles.rankNum}>{i + 1}</span>
                      </div>

                      {/* Body */}
                      <div style={styles.body}>
                        <div style={styles.row1}>
                          <h3 style={styles.movieTitle}>{m.title}</h3>
                          <span style={styles.pct}>{scoreToPercent(m.similarity_score)}%</span>
                        </div>

                        <div style={styles.metaRow}>
                          {m.director && <span style={styles.dir}>{m.director}</span>}
                          {m.director && formatYear(m.release_date) && <span style={styles.dot}>·</span>}
                          {formatYear(m.release_date) && <span style={styles.yr}>{formatYear(m.release_date)}</span>}
                          {m.avg_rating && <>
                            <span style={styles.dot}>·</span>
                            <span style={styles.rat}><span style={styles.star}>★</span> {m.avg_rating.toFixed(1)}</span>
                          </>}
                        </div>

                        {m.genres && (
                          <div style={styles.tags}>
                            {m.genres.slice(0, 4).map((g) => (
                              <span key={g} style={styles.tag}>{g}</span>
                            ))}
                          </div>
                        )}

                        <p style={styles.overview}>
                          {m.overview}
                        </p>

                        {m.cast && m.cast.length > 0 && (
                          <p style={styles.cast}>{m.cast.slice(0, 3).join("  ·  ")}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {/* Footer */}
        <div style={styles.footer}>
          <span style={styles.footTxt}>
            semantic retrieval · <span style={styles.footAcc}>12,083</span> films
          </span>
        </div>
      </div>

      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(20px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes drift1 {
          0%, 100% { transform: translate(0,0) scale(1); }
          33% { transform: translate(40px,-60px) scale(1.06); }
          66% { transform: translate(-25px,30px) scale(0.94); }
        }
        @keyframes drift2 {
          0%, 100% { transform: translate(0,0) scale(1); }
          33% { transform: translate(-50px,40px) scale(1.08); }
          66% { transform: translate(30px,-45px) scale(0.92); }
        }
        @keyframes drift3 {
          0%, 100% { transform: translate(0,0); }
          50% { transform: translate(30px,30px); }
        }
        @keyframes glowPulse {
          0%, 100% { opacity: 0.35; }
          50% { opacity: 0.75; }
        }
        input::placeholder { color: rgba(224,203,255,0.28); }
        input:focus { outline: none; }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { background: #1a1230; }
        body { overflow-x: hidden; }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(224,203,255,0.1); border-radius: 4px; }
      `}</style>
    </div>
  );
}

const styles = {
  root: {
    position: "relative",
    minHeight: "100vh",
    width: "100%",
    fontFamily: "'Outfit', sans-serif",
  },

  /* ── Background — brighter, with visible purple-teal hue ── */
  bgBase: {
    position: "fixed", inset: 0, zIndex: 0,
    background: "linear-gradient(155deg, #1e1640 0%, #1a1238 20%, #14162e 45%, #121a2a 65%, #181335 100%)",
  },
  bgOrb1: {
    position: "fixed", width: 800, height: 800, borderRadius: "50%", zIndex: 0,
    background: "radial-gradient(circle, rgba(168,85,247,0.18) 0%, rgba(139,92,246,0.06) 45%, transparent 70%)",
    top: "-18%", left: "-12%",
    animation: "drift1 28s ease-in-out infinite",
  },
  bgOrb2: {
    position: "fixed", width: 700, height: 700, borderRadius: "50%", zIndex: 0,
    background: "radial-gradient(circle, rgba(94,234,212,0.14) 0%, rgba(45,212,191,0.04) 45%, transparent 70%)",
    bottom: "-12%", right: "-10%",
    animation: "drift2 32s ease-in-out infinite",
  },
  bgOrb3: {
    position: "fixed", width: 450, height: 450, borderRadius: "50%", zIndex: 0,
    background: "radial-gradient(circle, rgba(192,132,252,0.12) 0%, transparent 70%)",
    top: "30%", right: "12%",
    animation: "drift3 22s ease-in-out infinite",
  },
  bgOrb4: {
    position: "fixed", width: 400, height: 400, borderRadius: "50%", zIndex: 0,
    background: "radial-gradient(circle, rgba(110,231,183,0.09) 0%, transparent 70%)",
    top: "55%", left: "8%",
    animation: "drift2 24s ease-in-out infinite reverse",
  },
  bgHaze: {
    position: "fixed", inset: 0, zIndex: 0,
    background: "radial-gradient(ellipse at 50% 40%, rgba(139,92,246,0.07) 0%, transparent 60%)",
  },
  noiseOverlay: {
    position: "fixed", inset: 0, zIndex: 1, pointerEvents: "none", opacity: 0.02,
    backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`,
    backgroundSize: "128px 128px",
  },

  /* ── Page shell ────────────────────────────── */
  page: {
    position: "relative", zIndex: 2,
    minHeight: "100vh",
    display: "flex", flexDirection: "column", alignItems: "center",
    padding: "0 28px",
  },

  /* ── Hero ───────────────────────────────────── */
  hero: {
    display: "flex", flexDirection: "column", alignItems: "center",
    width: "100%", maxWidth: 680,
    paddingTop: "26vh",
    transition: "padding-top 0.6s cubic-bezier(0.22, 1, 0.36, 1)",
  },
  heroCompact: {
    paddingTop: 40,
  },

  logo: {
    marginBottom: 28, cursor: "pointer",
    transition: "all 0.5s ease",
  },
  logoSmall: { marginBottom: 14 },

  titleWrap: {
    textAlign: "center",
    maxHeight: 160, opacity: 1,
    transition: "max-height 0.5s cubic-bezier(0.22,1,0.36,1), opacity 0.35s ease, margin 0.5s ease",
    overflow: "hidden",
    marginBottom: 10,
  },
  titleWrapHidden: {
    maxHeight: 0, opacity: 0, marginBottom: 0,
  },

  title: {
    fontFamily: "'Source Serif 4', serif",
    fontSize: 58, fontWeight: 300, fontStyle: "italic",
    letterSpacing: "-0.5px", lineHeight: 1.1,
    background: "linear-gradient(130deg, #e0cbff 0%, #c4b5fd 28%, #a5b4fc 55%, #6ee7b7 100%)",
    WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
    marginBottom: 16,
  },
  subtitle: {
    fontSize: 18, fontWeight: 300,
    color: "rgba(224,203,255,0.45)",
    letterSpacing: "0.2px", lineHeight: 1.5, marginBottom: 32,
  },

  /* ── Search ────────────────────────────────── */
  searchWrap: {
    position: "relative", width: "100%", marginBottom: 28,
    transition: "all 0.5s ease",
  },
  searchWrapCompact: { maxWidth: 600, marginBottom: 12 },

  searchGlow: {
    position: "absolute", inset: -1, borderRadius: 16,
    background: "linear-gradient(135deg, rgba(168,85,247,0.3), rgba(110,231,183,0.22))",
    filter: "blur(2px)",
    animation: "glowPulse 4s ease-in-out infinite",
  },
  searchBar: {
    position: "relative",
    display: "flex", alignItems: "center", gap: 14,
    padding: "17px 20px", borderRadius: 14,
    background: "rgba(18,14,34,0.75)",
    border: "1px solid rgba(224,203,255,0.13)",
    backdropFilter: "blur(24px)",
  },
  input: {
    flex: 1, background: "none", border: "none",
    color: "rgba(248,244,255,0.93)",
    fontSize: 17, fontFamily: "'Outfit', sans-serif", fontWeight: 300,
    letterSpacing: "0.2px", caretColor: "#c084fc",
  },
  clearBtn: {
    background: "none", border: "none",
    color: "rgba(224,203,255,0.35)", fontSize: 24,
    cursor: "pointer", padding: "0 4px", lineHeight: 1,
  },
  goBtn: {
    display: "flex", alignItems: "center", justifyContent: "center",
    width: 40, height: 40, borderRadius: 10, border: "none",
    background: "linear-gradient(135deg, rgba(168,85,247,0.4), rgba(110,231,183,0.28))",
    color: "rgba(248,244,255,0.95)", cursor: "pointer", flexShrink: 0,
    transition: "all 0.2s ease",
  },

  /* ── Chips ─────────────────────────────────── */
  chips: {
    display: "flex", flexWrap: "wrap", justifyContent: "center",
    gap: 10, maxHeight: 120, opacity: 1, overflow: "hidden",
    transition: "all 0.4s ease",
  },
  chipsHidden: { maxHeight: 0, opacity: 0, marginTop: 0 },

  chip: {
    padding: "9px 18px", borderRadius: 22,
    border: "1px solid rgba(224,203,255,0.15)",
    background: "rgba(224,203,255,0.06)",
    color: "rgba(224,203,255,0.5)",
    fontSize: 14, fontFamily: "'Outfit', sans-serif", fontWeight: 300,
    cursor: "pointer", transition: "all 0.2s ease", letterSpacing: "0.2px",
  },

  /* ── Results ───────────────────────────────── */
  resultsWrap: {
    width: "100%", maxWidth: 720,
    marginTop: 12, paddingBottom: 90,
  },
  resultsLabel: {
    fontSize: 14, fontWeight: 300,
    color: "rgba(224,203,255,0.32)",
    textAlign: "center", marginBottom: 24, letterSpacing: "0.3px",
  },
  labelQ: { color: "rgba(110,231,183,0.55)", fontStyle: "italic" },

  loadWrap: {
    display: "flex", flexDirection: "column", alignItems: "center",
    gap: 18, padding: "70px 0",
  },
  loadText: {
    fontSize: 15, fontWeight: 300,
    color: "rgba(224,203,255,0.3)", letterSpacing: "0.5px",
  },
  spinSm: {
    width: 16, height: 16,
    border: "1.5px solid rgba(224,203,255,0.15)",
    borderTopColor: "rgba(224,203,255,0.6)",
    borderRadius: "50%", animation: "spin 0.8s linear infinite",
  },
  spinLg: {
    width: 30, height: 30,
    border: "2px solid rgba(224,203,255,0.08)",
    borderTopColor: "rgba(224,203,255,0.5)",
    borderRadius: "50%", animation: "spin 0.8s linear infinite",
  },
  empty: {
    textAlign: "center", color: "rgba(224,203,255,0.3)",
    fontSize: 16, fontWeight: 300, padding: "70px 0",
  },

  /* ── Card list ─────────────────────────────── */
  list: { display: "flex", flexDirection: "column", gap: 10 },

  card: {
    display: "flex", gap: 18, padding: "28px 30px",
    borderRadius: 14,
    background: "rgba(22,16,42,0.55)",
    border: "1px solid rgba(224,203,255,0.08)",
    backdropFilter: "blur(10px)",
    transition: "all 0.25s cubic-bezier(0.22,1,0.36,1)",
    animation: "fadeUp 0.45s ease both",
    cursor: "default", alignItems: "flex-start",
  },

  rank: {
    width: 36, height: 36, borderRadius: 10,
    background: "linear-gradient(135deg, rgba(168,85,247,0.14), rgba(110,231,183,0.1))",
    display: "flex", alignItems: "center", justifyContent: "center",
    flexShrink: 0, marginTop: 2,
  },
  rankNum: {
    fontSize: 14, fontWeight: 600,
    color: "rgba(110,231,183,0.65)",
    fontFamily: "'Outfit', sans-serif",
  },

  body: { flex: 1, minWidth: 0 },

  row1: {
    display: "flex", justifyContent: "space-between",
    alignItems: "baseline", gap: 14, marginBottom: 5,
  },
  movieTitle: {
    fontFamily: "'Source Serif 4', serif",
    fontSize: 23, fontWeight: 400,
    color: "rgba(248,244,255,0.92)", lineHeight: 1.3,
  },
  pct: {
    fontSize: 13, fontWeight: 500,
    color: "rgba(110,231,183,0.5)",
    fontFamily: "'Outfit', sans-serif", flexShrink: 0,
    letterSpacing: "0.3px",
  },

  metaRow: {
    display: "flex", alignItems: "center", gap: 7,
    marginBottom: 9, flexWrap: "wrap",
  },
  dir: { fontSize: 15, fontWeight: 300, color: "rgba(224,203,255,0.5)" },
  yr:  { fontSize: 15, fontWeight: 300, color: "rgba(110,231,183,0.45)" },
  dot: { fontSize: 11, color: "rgba(224,203,255,0.2)" },
  rat: { fontSize: 15, fontWeight: 400, color: "rgba(248,244,255,0.6)" },
  star: { color: "rgba(250,204,21,0.7)", fontSize: 13 },

  tags: { display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 },
  tag: {
    fontSize: 12, fontWeight: 400,
    padding: "3px 11px", borderRadius: 7,
    background: "rgba(168,85,247,0.12)",
    color: "rgba(224,203,255,0.55)", letterSpacing: "0.2px",
  },

  overview: {
    fontSize: 15.5, fontWeight: 300,
    color: "rgba(224,203,255,0.4)",
    lineHeight: 1.7, marginBottom: 9,
  },
  cast: {
    fontSize: 13.5, fontWeight: 350,
    color: "rgba(224,203,255,0.25)", letterSpacing: "0.2px",
  },

  /* ── Footer ────────────────────────────────── */
  footer: {
    position: "fixed", bottom: 0, left: 0, right: 0,
    padding: 20, textAlign: "center", zIndex: 10,
    background: "linear-gradient(transparent, rgba(22,16,42,0.95))",
    pointerEvents: "none",
  },
  footTxt: {
    fontSize: 11, fontWeight: 300,
    color: "rgba(224,203,255,0.2)",
    letterSpacing: 2, textTransform: "uppercase",
  },
  footAcc: { color: "rgba(110,231,183,0.4)" },
};
