import { useContext, useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  BarChart3,
  Check,
  ChevronDown,
  Compass,
  Menu,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";
import {
  animate,
  AnimatePresence,
  motion,
  useInView,
  useMotionValue,
  useReducedMotion,
  useScroll,
  useTransform,
} from "motion/react";
import type { EPLDataset, Fixture } from "../../types";
import {
  MotionPreferenceContext,
} from "../Motion";
import { dashboardRoutes, type AppRoute } from "../../lib/appRoute";
import { getLandingData } from "../../lib/landingData";
import "../../styles/landing.css";

type LandingPageProps = {
  dataset?: EPLDataset | null;
  onNavigate: (route: AppRoute) => void;
};

const landingLinks = [
  { label: "Features", href: "#features" },
  { label: "Methodology", href: "#methodology" },
  { label: "Trust", href: "#trust" },
];

const dashboardLabels: Record<Exclude<AppRoute, "landing">, string> = {
  fixtures: "Fixtures",
  simulator: "Simulator",
  standings: "Table",
  clubs: "Clubs",
  analytics: "Analytics",
};

const reveal = {
  hidden: { opacity: 0, y: 24 },
  visible: { opacity: 1, y: 0 },
};

const stagger = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.08, delayChildren: 0.12 } },
};

const featureCards = [
  {
    icon: BarChart3,
    eyebrow: "01 / Forecasts",
    title: "Read the match before kickoff.",
    body: "Calibrated Home, Draw, and Away probabilities sit beside the scoreline, expected goals, and the signal that moved the model.",
  },
  {
    icon: Compass,
    eyebrow: "02 / Scenarios",
    title: "Explore the assumptions.",
    body: "Change the matchup, adjust the context, and understand how small shifts in attack, defence, or venue alter the outlook.",
  },
  {
    icon: ShieldCheck,
    eyebrow: "03 / Evidence",
    title: "Trust the process, not the hype.",
    body: "Time-ordered validation, leakage-safe features, and transparent diagnostics make every forecast easier to interrogate.",
  },
];

const getProbability = (fixture: Fixture | undefined, key: "homeWinProb" | "drawProb" | "awayWinProb") =>
  fixture ? `${fixture[key].toFixed(1)}%` : "—";

const Counter: React.FC<{ value: number; suffix?: string; label: string }> = ({
  value,
  suffix = "",
  label,
}) => {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-15%" });
  const motionValue = useMotionValue(0);
  const rounded = useTransform(motionValue, (current) => Math.round(current).toString());
  const reduceMotion = useReducedMotion() || useContext(MotionPreferenceContext);

  useEffect(() => {
    if (!inView) return undefined;
    if (reduceMotion) {
      motionValue.set(value);
      return undefined;
    }
    const controls = animate(motionValue, value, { duration: 1.1, ease: [0.22, 1, 0.36, 1] });
    return () => controls.stop();
  }, [inView, motionValue, reduceMotion, value]);

  return (
    <div ref={ref} className="landing-stat">
      <strong><motion.span>{rounded}</motion.span>{suffix}</strong>
      <span>{label}</span>
    </div>
  );
};

const LandingNav: React.FC<Pick<LandingPageProps, "onNavigate">> = ({ onNavigate }) => {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const reduceMotion = useReducedMotion();

  useEffect(() => {
    if (!menuOpen) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenuOpen(false);
        menuButtonRef.current?.focus();
        return;
      }
      if (event.key !== "Tab" || !menuRef.current) return;
      const focusable = menuRef.current.querySelectorAll<HTMLElement>("a, button");
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);
    menuRef.current?.querySelector<HTMLElement>("a")?.focus();
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [menuOpen]);

  const scrollTo = (href: string) => {
    setMenuOpen(false);
    document.querySelector(href)?.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
  };

  return (
    <header className="landing-nav-wrap">
      <nav className="landing-nav" aria-label="Landing page navigation">
        <button className="landing-brand" type="button" onClick={() => scrollTo("#top")} aria-label="Return to top — EPL Predictor">
          <img
            className="landing-brand-logo"
            src={`${import.meta.env.BASE_URL}epl-predictor-header-logo.png`}
            alt="EPL Predictor"
          />
          <span className="landing-brand-season"><small>2026 / 27</small></span>
        </button>
        <div className="landing-nav-links">
          {landingLinks.map((link) => (
            <a key={link.href} href={link.href} onClick={(event) => { event.preventDefault(); scrollTo(link.href); }}>
              {link.label}
            </a>
          ))}
        </div>
        <button className="landing-nav-cta" type="button" onClick={() => onNavigate("fixtures")}>
          Open dashboard <ArrowRight size={15} aria-hidden="true" />
        </button>
        <button
          ref={menuButtonRef}
          className="landing-menu-button"
          type="button"
          aria-expanded={menuOpen}
          aria-controls="landing-mobile-menu"
          aria-label={menuOpen ? "Close navigation menu" : "Open navigation menu"}
          onClick={() => setMenuOpen((open) => !open)}
        >
          <AnimatePresence mode="wait" initial={false}>
            <motion.span key={menuOpen ? "close" : "open"} initial={{ rotate: -45, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: 45, opacity: 0 }}>
              {menuOpen ? <X size={21} aria-hidden="true" /> : <Menu size={21} aria-hidden="true" />}
            </motion.span>
          </AnimatePresence>
        </button>
      </nav>
      <AnimatePresence>
        {menuOpen && (
          <motion.div
            id="landing-mobile-menu"
            ref={menuRef}
            className="landing-mobile-menu"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: reduceMotion ? 0 : 0.22 }}
          >
            {landingLinks.map((link, index) => (
              <motion.a
                key={link.href}
                href={link.href}
                initial={reduceMotion ? false : { opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: reduceMotion ? 0 : index * 0.05 }}
                onClick={(event) => { event.preventDefault(); scrollTo(link.href); }}
              >
                <span>0{index + 1}</span>{link.label}<ArrowRight size={15} aria-hidden="true" />
              </motion.a>
            ))}
            <span className="landing-mobile-menu-label">Dashboard views</span>
            {dashboardRoutes.map((route, index) => (
              <motion.button
                key={route}
                type="button"
                initial={reduceMotion ? false : { opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: reduceMotion ? 0 : (landingLinks.length + index) * 0.05 }}
                onClick={() => { setMenuOpen(false); onNavigate(route); }}
              >
                <span>0{landingLinks.length + index + 1}</span>{dashboardLabels[route]}<ArrowRight size={15} aria-hidden="true" />
              </motion.button>
            ))}
            <button type="button" onClick={() => { setMenuOpen(false); onNavigate("fixtures"); }}>
              Open dashboard <ArrowRight size={15} aria-hidden="true" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
};

export const LandingPage: React.FC<LandingPageProps> = ({ dataset, onNavigate }) => {
  const heroVisualRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: heroVisualRef, offset: ["start end", "end start"] });
  const visualY = useTransform(scrollYProgress, [0, 1], [18, -18]);
  const visualScale = useTransform(scrollYProgress, [0, 0.5, 1], [0.96, 1, 0.98]);
  const reduceMotion = useReducedMotion() || useContext(MotionPreferenceContext);
  const landingData = getLandingData(dataset);
  const fixture = landingData.fixture;
  const playedMatches = landingData.playedMatches;

  return (
    <MotionPreferenceContext.Provider value={Boolean(reduceMotion)}>
      <div className="landing-page" id="top">
        <LandingNav onNavigate={onNavigate} />
        <main>
          <section className="landing-hero" aria-labelledby="landing-title">
            <motion.div className="landing-hero-copy" variants={stagger} initial={reduceMotion ? false : "hidden"} animate="visible">
              <motion.div className="landing-kicker" variants={reveal}><span className="status-dot" /> Premier League / Match intelligence</motion.div>
              <motion.h1 id="landing-title" variants={reveal}>See the game<br /><em>before it starts.</em></motion.h1>
              <motion.p className="landing-hero-lede" variants={reveal}>A calm, evidence-led way to read every fixture. Calibrated probabilities, expected goals, and the context behind the call.</motion.p>
              <motion.div className="landing-hero-actions" variants={reveal}>
                <button className="landing-primary-button" type="button" onClick={() => onNavigate("fixtures")}>
                  Explore forecasts <ArrowRight size={17} aria-hidden="true" />
                </button>
                <a className="landing-text-link" href="#methodology" onClick={(event) => { event.preventDefault(); document.querySelector("#methodology")?.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" }); }}>
                  How it works <ChevronDown size={16} aria-hidden="true" />
                </a>
              </motion.div>
              <motion.div className="landing-hero-note" variants={reveal}><Check size={14} aria-hidden="true" /> No live odds API. No black box theatre.</motion.div>
            </motion.div>
            <motion.div ref={heroVisualRef} className="landing-hero-visual-wrap" style={reduceMotion ? undefined : { y: visualY, scale: visualScale }} initial={reduceMotion ? false : { opacity: 0, scale: 0.94, y: 20 }} animate={{ opacity: 1, scale: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}>
              <div className="landing-hero-orbit orbit-one" />
              <div className="landing-hero-orbit orbit-two" />
              <div className="landing-forecast-card">
                <div className="landing-card-topline"><span>Next signal</span><span>{fixture?.gameweek ? `GW ${fixture.gameweek}` : "2026 / 27"}</span></div>
                <div className="landing-card-matchup">
                  <div><span className="landing-team-crest">{fixture?.homeTeam?.slice(0, 1) ?? "H"}</span><strong>{fixture?.homeTeam ?? "Home side"}</strong><small>Home</small></div>
                  <div className="landing-card-score"><span>{fixture?.predictedScore ?? "—"}</span><small>model score</small></div>
                  <div><span className="landing-team-crest away">{fixture?.awayTeam?.slice(0, 1) ?? "A"}</span><strong>{fixture?.awayTeam ?? "Away side"}</strong><small>Away</small></div>
                </div>
                <div className="landing-probability-label"><span>Probability field</span><strong>{fixture ? `${Math.max(fixture.homeWinProb, fixture.drawProb, fixture.awayWinProb).toFixed(1)}% strongest signal` : "Awaiting data"}</strong></div>
                <div className="landing-probability-bars">
                  {[{ label: "Home", value: getProbability(fixture, "homeWinProb"), width: fixture?.homeWinProb ?? 0, className: "home" }, { label: "Draw", value: getProbability(fixture, "drawProb"), width: fixture?.drawProb ?? 0, className: "draw" }, { label: "Away", value: getProbability(fixture, "awayWinProb"), width: fixture?.awayWinProb ?? 0, className: "away" }].map((item) => (
                    <div key={item.label} className="landing-probability-row"><span>{item.label}</span><div><motion.i initial={{ width: 0 }} animate={{ width: `${item.width}%` }} transition={{ duration: 0.8, delay: 0.65 }} className={item.className} /></div><strong>{item.value}</strong></div>
                  ))}
                </div>
                <div className="landing-card-foot"><span className="status-dot" /> Calibrated / offline dataset <ArrowRight size={14} aria-hidden="true" /></div>
              </div>
              <div className="landing-floating-label label-top"><Sparkles size={14} aria-hidden="true" /> Form + market context</div>
              <div className="landing-floating-label label-bottom"><span>RPS</span> {landingData.productionRps.toFixed(4)} <small>{landingData.productionModel} · best in benchmark</small></div>
            </motion.div>
          </section>

          <section className="landing-trust-strip" id="trust" aria-label="Platform facts">
            <div className="landing-section-label">Built for better questions</div>
            <div className="landing-stats">
              <Counter value={landingData.totalMatches} label="fixture season slate" />
              <Counter value={3} label="outcome probabilities" />
              <Counter value={playedMatches} label="results recorded" />
              <div className="landing-stat landing-stat-text"><strong>0<span>ms</span></strong><span>live odds dependency</span></div>
            </div>
          </section>

          <section className="landing-section landing-features" id="features" aria-labelledby="features-title">
            <motion.div className="landing-section-heading" initial={reduceMotion ? false : "hidden"} whileInView="visible" viewport={{ once: true, amount: 0.3 }} variants={stagger}>
              <motion.div className="landing-section-label" variants={reveal}>The workspace</motion.div>
              <motion.h2 id="features-title" variants={reveal}>Prediction, with<br /><em>the why included.</em></motion.h2>
              <motion.p variants={reveal}>Designed for the moment between a fixture announcement and a final opinion.</motion.p>
            </motion.div>
            <div className="landing-feature-grid">
              {featureCards.map((feature, index) => {
                const Icon = feature.icon;
                return <motion.article key={feature.eyebrow} className="landing-feature-card" initial={reduceMotion ? false : "hidden"} whileInView="visible" whileHover={reduceMotion ? undefined : { y: -6 }} whileTap={reduceMotion ? undefined : { scale: 0.985 }} viewport={{ once: true, amount: 0.2 }} variants={reveal} transition={{ duration: 0.45, delay: index * 0.08 }}>
                  <div className="landing-feature-icon"><Icon size={20} strokeWidth={1.7} aria-hidden="true" /></div>
                  <span className="landing-feature-eyebrow">{feature.eyebrow}</span>
                  <h3>{feature.title}</h3>
                  <p>{feature.body}</p>
                  <span className="landing-feature-arrow"><ArrowRight size={16} aria-hidden="true" /></span>
                </motion.article>;
              })}
            </div>
          </section>

          <section className="landing-methodology" id="methodology" aria-labelledby="methodology-title">
            <div className="landing-methodology-graphic" aria-hidden="true"><div className="methodology-line line-a" /><div className="methodology-line line-b" /><span className="methodology-node node-a" /><span className="methodology-node node-b" /><span className="methodology-node node-c" /></div>
            <motion.div className="landing-methodology-copy" initial={reduceMotion ? false : "hidden"} whileInView="visible" viewport={{ once: true, amount: 0.35 }} variants={stagger}>
              <motion.div className="landing-section-label" variants={reveal}>A clear signal chain</motion.div>
              <motion.h2 id="methodology-title" variants={reveal}>From history<br />to <em>confidence.</em></motion.h2>
              <motion.p variants={reveal}>The model learns chronologically, tests itself honestly, and explains its forecast in the language of the match.</motion.p>
              <motion.ol className="landing-process-list" variants={stagger}>
                {["Shape the context", "Benchmark the models", "Publish the signal"].map((step, index) => <motion.li key={step} variants={reveal}><span>0{index + 1}</span><strong>{step}</strong><small>{["Form, venue, rest, Elo, and pre-kickoff market context.", "Time-ordered validation selects the most reliable ensemble.", "A readable probability field, scoreline, and evidence state."][index]}</small></motion.li>)}
              </motion.ol>
            </motion.div>
          </section>

          <section className="landing-final-cta" aria-labelledby="final-cta-title">
            <motion.div initial={reduceMotion ? false : { opacity: 0, y: 18 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.5 }}>
              <span className="landing-section-label">Your next fixture</span>
              <h2 id="final-cta-title">Make the call<br /><em>with context.</em></h2>
              <button className="landing-primary-button" type="button" onClick={() => onNavigate("fixtures")}>Enter the workspace <ArrowRight size={17} aria-hidden="true" /></button>
            </motion.div>
          </section>
        </main>
        <footer className="landing-footer">
          <span>Forecasts are probabilities, not guarantees.</span>
          <div><button type="button" onClick={() => onNavigate("fixtures")}>Dashboard</button><a href="https://github.com/Raghav2012Code/epl-predictor" target="_blank" rel="noreferrer">Source <ArrowRight size={13} aria-hidden="true" /></a></div>
          <span>© {new Date().getFullYear()} EPL Predictor</span>
        </footer>
      </div>
    </MotionPreferenceContext.Provider>
  );
};
