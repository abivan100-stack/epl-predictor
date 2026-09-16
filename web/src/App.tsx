import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  Search,
  X,
} from "lucide-react";
import { EPLDataset, Fixture, TeamProfile } from "./types";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { AppSkeleton, DataErrorPanel } from "./components/DataStates";
import { useEPLData } from "./hooks/useEPLData";
import {
  MotionItem,
  MotionList,
  MotionPreferenceProvider,
  MotionSection,
} from "./components/Motion";
import { Grid } from "./components/charts/grid";
import { ChartTooltip } from "./components/charts/tooltip";
import { LineChart, Line } from "./components/charts/line-chart";
import { XAxis } from "./components/charts/x-axis";
import { RingChart } from "./components/charts/ring-chart";
import { Ring } from "./components/charts/ring";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { LandingPage } from "./components/landing/LandingPage";
import { appRouteForPath, pathForAppRoute, type AppRoute } from "./lib/appRoute";

type Tab = "fixtures" | "simulator" | "standings" | "clubs" | "analytics";
type SortKey =
  "rank" | "team" | "played" | "won" | "drawn" | "lost" | "gd" | "points";
type FixtureFilter = "all" | "upcoming" | "played";
type FixtureSort = "date" | "confidence";
const tabs: Array<{ id: Tab; label: string; note: string }> = [
  { id: "fixtures", label: "Fixtures", note: "Gameweeks and forecasts" },
  { id: "simulator", label: "Simulator", note: "Explore a matchup" },
  { id: "standings", label: "Table", note: "Current and projected table" },
  { id: "clubs", label: "Clubs", note: "Team profiles" },
  { id: "analytics", label: "Analytics", note: "Model evidence and trends" },
];
const pct = (value: number) => `${Number(value).toFixed(1)}%`;
const confidenceFor = (fixture: Fixture) =>
  Math.max(fixture.homeWinProb, fixture.drawProb, fixture.awayWinProb);
const deltaLabel = (value: number) =>
  `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
const downloadFixturesCsv = (fixtures: Fixture[]) => {
  const rows = [
    ["Date", "Gameweek", "Home", "Away", "Prediction", "Outcome", "Confidence"],
    ...fixtures.map((fixture) => [
      fixture.date,
      String(fixture.gameweek),
      fixture.homeTeam,
      fixture.awayTeam,
      fixture.predictedScore,
      fixture.predictedOutcome,
      pct(confidenceFor(fixture)),
    ]),
  ];
  const csv = rows.map((row) => row.map((cell) => `"${cell.replaceAll('"', '""')}"`).join(",")).join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "epl-predictor-fixtures.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
};
type ScenarioResult = {
  homeExpected: number;
  awayExpected: number;
  homeProb: number;
  drawProb: number;
  awayProb: number;
  homeScore: number;
  awayScore: number;
};
const calculateScenario = (
  home: TeamProfile | undefined,
  away: TeamProfile | undefined,
  homeBoost: number,
  awayBoost: number,
  neutral: boolean,
): ScenarioResult | null => {
  if (!home || !away) return null;
  const homeAttack = Math.max(0.2, home.gfPerMatch * (1 + homeBoost / 100));
  const awayAttack = Math.max(0.2, away.gfPerMatch * (1 + awayBoost / 100));
  const homeExpected = Math.max(
    0.2,
    ((homeAttack + away.gaPerMatch) / 2) * (neutral ? 1 : 1.18),
  );
  const awayExpected = Math.max(
    0.15,
    ((awayAttack + home.gaPerMatch) / 2) * (neutral ? 1 : 0.9),
  );
  const denominator = Math.exp(homeExpected) + Math.exp(awayExpected) + 1.2;
  const homeProb = Math.round((Math.exp(homeExpected) / denominator) * 1000) / 10;
  const drawProb = Math.round(
    (1 - (Math.exp(homeExpected) + Math.exp(awayExpected)) / denominator) * 1000,
  ) / 10;
  const awayProb = Math.round((100 - homeProb - drawProb) * 10) / 10;
  return { homeExpected, awayExpected, homeProb, drawProb, awayProb, homeScore: Math.round(homeExpected), awayScore: Math.round(awayExpected) };
};
const displayDate = (value: string) =>
  new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  }).format(new Date(`${value}T12:00:00`));
const scoreParts = (score: string): [number, number] | null => {
  const match = score.match(/(\d+)\s*[-–]\s*(\d+)/);
  return match ? [Number(match[1]), Number(match[2])] : null;
};
const startOfToday = () => {
  const day = new Date();
  day.setHours(0, 0, 0, 0);
  return day.getTime();
};
const fixtureDay = (value: string) => new Date(`${value}T00:00:00`).getTime();
const nextFixtureByDate = (fixtures: Fixture[]) =>
  [...fixtures]
    .filter(
      (fixture) =>
        fixture.status !== "Played" &&
        fixtureDay(fixture.date) >= startOfToday(),
    )
    .sort(
      (a, b) =>
        fixtureDay(a.date) - fixtureDay(b.date) ||
        a.gameweek - b.gameweek ||
        a.id - b.id,
    )[0] ?? null;
const resultFor = (fixture: Fixture) => {
  const score = scoreParts(fixture.actualScore);
  if (!score) return null;
  return score[0] === score[1]
    ? "Draw"
    : score[0] > score[1]
      ? "Home win"
      : "Away win";
};
const clubResultFor = (fixture: Fixture, club: string) => {
  const result = resultFor(fixture);
  if (!result) return null;
  if (result === "Draw") return "Draw";
  const clubWasHome = fixture.homeTeam === club;
  const clubWon = result === "Home win" ? clubWasHome : !clubWasHome;
  return clubWon ? "Win" : "Loss";
};
const resultTone = (result: string | null) =>
  result === "Home win" || result === "Win"
    ? "tone-win"
    : result === "Away win" || result === "Loss"
      ? "tone-loss"
      : "tone-draw";

const TeamMark: React.FC<{
  team?: TeamProfile;
  short?: string;
  badge?: string;
}> = ({ team, short, badge }) => {
  const src = badge ?? team?.badge ?? "";
  const [failed, setFailed] = useState(false);
  if (src && !failed)
    return (
      <img
        className="team-mark team-badge"
        src={`${import.meta.env.BASE_URL}${src}`}
        alt=""
        aria-hidden="true"
        loading="lazy"
        draggable={false}
        onError={() => setFailed(true)}
      />
    );
  return (
    <span
      className="team-mark"
      style={{ borderColor: team?.color ?? "#1d6f52" }}
      aria-hidden="true"
    >
      {(short ?? team?.short ?? "FC").slice(0, 3)}
    </span>
  );
};
const ProbabilityStrip: React.FC<{ fixture: Fixture }> = ({ fixture }) => (
  <div
    className="probability-strip"
    aria-label={`Home ${pct(fixture.homeWinProb)}, draw ${pct(fixture.drawProb)}, away ${pct(fixture.awayWinProb)}`}
  >
    <span className="prob-home" style={{ width: `${fixture.homeWinProb}%` }} />
    <span className="prob-draw" style={{ width: `${fixture.drawProb}%` }} />
    <span className="prob-away" style={{ width: `${fixture.awayWinProb}%` }} />
  </div>
);

const FixtureRow: React.FC<{
  fixture: Fixture;
  selected: boolean;
  onSelect: () => void;
}> = ({ fixture, selected, onSelect }) => {
  const actual = fixture.status === "Played" ? fixture.actualScore : null;
  return (
    <button
      className={`fixture-row ${selected ? "is-selected" : ""}`}
      onClick={onSelect}
      aria-pressed={selected}
    >
      <span className="fixture-date">
        {displayDate(fixture.date)}
        <small>
          GW{fixture.gameweek} ·{" "}
          {fixture.time === "TBC" ? "Kickoff TBC" : fixture.time}
        </small>
      </span>
      <span className="fixture-teams">
        <span>
          <TeamMark short={fixture.homeShort} badge={fixture.homeBadge} />
          {fixture.homeTeam}
        </span>
        <span>
          <TeamMark short={fixture.awayShort} badge={fixture.awayBadge} />
          {fixture.awayTeam}
        </span>
      </span>
      <span className="fixture-score">
        <small>{actual ? "Final" : "Model score"}</small>
        {actual ?? fixture.predictedScore}
      </span>
      <span className="fixture-favorite">
        <small>
          {fixture.status === "Played"
            ? resultFor(fixture)
            : fixture.predictedOutcome}
        </small>
        {fixture.status === "Played"
          ? "Result logged"
          : `${pct(Math.max(fixture.homeWinProb, fixture.drawProb, fixture.awayWinProb))} strongest signal`}
      </span>
      <ArrowUpRight className="row-arrow" size={16} aria-hidden="true" />
    </button>
  );
};

const FixtureDetail: React.FC<{
  dataset: EPLDataset;
  fixture: Fixture | null;
  onSimulate: (home: string, away: string) => void;
}> = ({ dataset, fixture, onSimulate }) => {
  if (!fixture)
    return (
      <div className="detail-panel empty-state">
        Choose a fixture to inspect it.
      </div>
    );
  const isPlayed = fixture.status === "Played";
  const homeProfile = dataset.teams[fixture.homeTeam];
  const awayProfile = dataset.teams[fixture.awayTeam];
  return (
    <article className="detail-panel">
      <div className="detail-top">
        <span
          className={`status-pill ${isPlayed ? "status-played" : "status-upcoming"}`}
        >
          {isPlayed ? "Final result" : "Forecast"}
        </span>
        <span>
          GW{fixture.gameweek} · {displayDate(fixture.date)} ·{" "}
          {fixture.time === "TBC" ? "Kickoff TBC" : fixture.time}
        </span>
      </div>
      <div className="matchup">
        <div>
          <TeamMark short={fixture.homeShort} badge={fixture.homeBadge} />
          <strong>{fixture.homeTeam}</strong>
          <small>Home</small>
        </div>
        <div className="matchup-score">
          <span>{isPlayed ? fixture.actualScore : fixture.predictedScore}</span>
          <small>{isPlayed ? "official score" : "most likely scoreline"}</small>
          {isPlayed && <small>Model had {fixture.predictedScore}</small>}
        </div>
        <div>
          <TeamMark short={fixture.awayShort} badge={fixture.awayBadge} />
          <strong>{fixture.awayTeam}</strong>
          <small>Away</small>
        </div>
      </div>
      <ProbabilityStrip fixture={fixture} />
      <div className="probability-labels">
        <span>
          <b>{pct(fixture.homeWinProb)}</b> Home
        </span>
        <span>
          <b>{pct(fixture.drawProb)}</b> Draw
        </span>
        <span>
          <b>{pct(fixture.awayWinProb)}</b> Away
        </span>
      </div>
      <div className="detail-metrics" aria-label="Forecast summary">
        <div>
          <span>Expected goals</span>
          <strong>{fixture.predHomeGoals.toFixed(2)} — {fixture.predAwayGoals.toFixed(2)}</strong>
        </div>
        <div>
          <span>Strongest signal</span>
          <strong>{pct(confidenceFor(fixture))}</strong>
        </div>
        <div>
          <span>Evidence state</span>
          <strong>{isPlayed ? "Measured" : "Projected"}</strong>
        </div>
      </div>
      <div className="form-strip" aria-label="Team context">
        <div><span>{fixture.homeTeam} form</span><strong>{homeProfile?.last5Form?.join(" ") ?? "Unavailable"}</strong><small>{homeProfile?.restDaysAvg ?? "—"}d average rest</small></div>
        <div><span>{fixture.awayTeam} form</span><strong>{awayProfile?.last5Form?.join(" ") ?? "Unavailable"}</strong><small>{awayProfile?.restDaysAvg ?? "—"}d average rest</small></div>
      </div>
      <div className="detail-copy">
        <p>
          <strong>{isPlayed ? "Model review" : "Model read"}</strong>{" "}
          {isPlayed
            ? `The model selected ${fixture.predictedOutcome.toLowerCase()} and the official result was ${fixture.actualScore}.`
            : `The model leans ${fixture.predictedOutcome.toLowerCase()} with a ${fixture.predictedScore} scoreline.`}
        </p>
        <p className="muted">
          Percentages are rounded to one decimal place and always total 100%.
          Exact scores are illustrative Poisson modes, not certainties.
        </p>
      </div>
      <button
        className="primary-button"
        onClick={() => onSimulate(fixture.homeTeam, fixture.awayTeam)}
      >
        Open in simulator <ArrowUpRight size={16} />
      </button>
    </article>
  );
};

const FixturesPage: React.FC<{
  dataset: EPLDataset;
  query: string;
  onSimulate: (home: string, away: string) => void;
}> = ({ dataset, query, onSimulate }) => {
  const gameweeks = useMemo(
    () =>
      [...new Set(dataset.fixtures.map((fixture) => fixture.gameweek))].sort(
        (a, b) => a - b,
      ),
    [dataset.fixtures],
  );
  const firstUpcoming =
    dataset.fixtures.find((fixture) => fixture.status !== "Played")?.gameweek ??
    1;
  const autoFixture = useMemo(
    () => nextFixtureByDate(dataset.fixtures),
    [dataset.fixtures],
  );
  const [gameweek, setGameweek] = useState(
    autoFixture?.gameweek ?? firstUpcoming,
  );
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [filter, setFilter] = useState<FixtureFilter>("all");
  const [sort, setSort] = useState<FixtureSort>("date");
  const filtered = useMemo(
    () =>
      dataset.fixtures.filter(
        (fixture) =>
          fixture.gameweek === gameweek &&
          (!query ||
            `${fixture.homeTeam} ${fixture.awayTeam}`
              .toLowerCase()
              .includes(query.toLowerCase())),
      ),
    [dataset.fixtures, gameweek, query],
  );
  const visibleFixtures = useMemo(() => {
    const matchesFilter = filtered.filter((fixture) =>
      filter === "all" ? true : filter === "played" ? fixture.status === "Played" : fixture.status !== "Played",
    );
    return [...matchesFilter].sort((a, b) =>
      sort === "confidence"
        ? confidenceFor(b) - confidenceFor(a) || a.id - b.id
        : fixtureDay(a.date) - fixtureDay(b.date) || a.id - b.id,
    );
  }, [filter, filtered, sort]);
  useEffect(() => {
    const upcoming = visibleFixtures
      .filter(
        (fixture) =>
          fixture.status !== "Played" &&
          fixtureDay(fixture.date) >= startOfToday(),
      )
      .sort(
        (a, b) => fixtureDay(a.date) - fixtureDay(b.date) || a.id - b.id,
      )[0];
    setSelectedId((upcoming ?? visibleFixtures[0])?.id ?? null);
  }, [gameweek, query, visibleFixtures]);
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.isContentEditable || ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
      if (event.key === "ArrowLeft") setGameweek((current) => Math.max(gameweeks[0] ?? 1, current - 1));
      if (event.key === "ArrowRight") setGameweek((current) => Math.min(gameweeks[gameweeks.length - 1] ?? 38, current + 1));
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [gameweeks]);
  const selected =
    visibleFixtures.find((fixture) => fixture.id === selectedId) ??
    visibleFixtures[0] ??
    null;
  const played = dataset.fixtures.filter(
    (fixture) => fixture.status === "Played",
  ).length;
  const model =
    dataset.benchmark.models.find((entry) => entry.isProduction) ??
    dataset.benchmark.models[0];
  return (
    <MotionSection className="page-stack">
      <div className="page-intro">
        <div>
          <p className="eyebrow">2026/27 season workspace</p>
          <h1>Fixtures, with the model beside them.</h1>
          <p className="lede">
            Browse each gameweek, see the forecast in plain language, and open a
            matchup when you want to inspect the assumptions.
          </p>
        </div>
        <div className="intro-stat">
          <strong>
            {played}/{dataset.totalMatches}
          </strong>
          <span>results recorded</span>
        </div>
      </div>
      <div className="stat-grid">
        <div>
          <span>Production model</span>
          <strong>{dataset.benchmark.productionModel}</strong>
        </div>
        <div>
          <span>Validation accuracy</span>
          <strong>{model?.accuracy ?? "—"}%</strong>
        </div>
        <div>
          <span>Average goal error</span>
          <strong>{model?.avgGoalMae ?? "—"}</strong>
        </div>
        <div>
          <span>Data basis</span>
          <strong>Time series</strong>
        </div>
      </div>
      <div className="section-heading">
        <div>
          <p className="eyebrow">Schedule</p>
          <h2>Gameweek {gameweek}</h2>
        </div>
        <div className="stepper">
          <button
            onClick={() =>
              setGameweek((current) => Math.max(gameweeks[0] ?? 1, current - 1))
            }
            disabled={gameweek <= (gameweeks[0] ?? 1)}
            aria-label="Previous gameweek"
          >
            <ChevronLeft size={18} />
          </button>
          <select
            value={gameweek}
            onChange={(event) => setGameweek(Number(event.target.value))}
            aria-label="Select gameweek"
          >
            {gameweeks.map((gw) => (
              <option key={gw} value={gw}>
                Gameweek {gw}
              </option>
            ))}
          </select>
          <button
            onClick={() =>
              setGameweek((current) =>
                Math.min(gameweeks[gameweeks.length - 1] ?? 38, current + 1),
              )
            }
            disabled={gameweek >= (gameweeks[gameweeks.length - 1] ?? 38)}
            aria-label="Next gameweek"
          >
            <ChevronRight size={18} />
          </button>
        </div>
      </div>
      <div className="fixture-controls" aria-label="Fixture filters">
        <div className="filter-group" role="group" aria-label="Fixture status">
          {([['all', 'All'], ['upcoming', 'Upcoming'], ['played', 'Played']] as const).map(([value, label]) => (
            <button key={value} className={filter === value ? "active" : ""} aria-pressed={filter === value} onClick={() => setFilter(value)}>{label}</button>
          ))}
        </div>
        <label className="sort-control">Sort by
          <select value={sort} onChange={(event) => setSort(event.target.value as FixtureSort)} aria-label="Sort fixtures">
            <option value="date">Date</option>
            <option value="confidence">Confidence</option>
          </select>
        </label>
        <span className="muted">{visibleFixtures.length} of {filtered.length} fixtures</span>
        <div className="fixture-actions">
          <button className="text-button" onClick={() => downloadFixturesCsv(visibleFixtures)}>Export CSV</button>
          <button className="text-button" onClick={() => window.print()}>Print</button>
        </div>
      </div>
      <div className="fixture-layout">
        <div
          className="fixture-list"
          aria-label={`Gameweek ${gameweek} fixtures`}
        >
          {visibleFixtures.length ? (
            <MotionList className="fixture-motion-list">
              {visibleFixtures.map((fixture) => (
                <MotionItem key={fixture.id} className="fixture-motion-item">
                  <FixtureRow
                    fixture={fixture}
                    selected={fixture.id === selected?.id}
                    onSelect={() => setSelectedId(fixture.id)}
                  />
                </MotionItem>
              ))}
            </MotionList>
          ) : (
            <div className="empty-state">
              No fixtures match this gameweek and search.
            </div>
          )}
        </div>
        <FixtureDetail dataset={dataset} fixture={selected} onSimulate={onSimulate} />
      </div>
    </MotionSection>
  );
};

const SimulatorPage: React.FC<{
  dataset: EPLDataset;
  initialHome: string;
  initialAway: string;
}> = ({ dataset, initialHome, initialAway }) => {
  const names = useMemo(
    () => Object.keys(dataset.teams).sort(),
    [dataset.teams],
  );
  const [homeTeam, setHomeTeam] = useState(initialHome);
  const [awayTeam, setAwayTeam] = useState(initialAway);
  const [homeBoost, setHomeBoost] = useState(0);
  const [awayBoost, setAwayBoost] = useState(0);
  const [neutral, setNeutral] = useState(false);
  useEffect(() => {
    setHomeTeam(initialHome);
    setAwayTeam(initialAway);
  }, [initialHome, initialAway]);
  const home = dataset.teams[homeTeam] ?? dataset.teams[names[0]];
  const away = dataset.teams[awayTeam] ?? dataset.teams[names[1]];
  const production = useMemo(() => {
    const direct = dataset.fixtures.find(
      (fixture) =>
        fixture.homeTeam === homeTeam && fixture.awayTeam === awayTeam,
    );
    if (direct) return direct;
    const reverse = dataset.fixtures.find(
      (fixture) =>
        fixture.homeTeam === awayTeam && fixture.awayTeam === homeTeam,
    );
    if (!reverse) return null;
    const score = scoreParts(reverse.predictedScore);
    return {
      ...reverse,
      homeTeam,
      awayTeam,
      homeShort: reverse.awayShort,
      awayShort: reverse.homeShort,
      homeColor: reverse.awayColor,
      awayColor: reverse.homeColor,
      predictedScore: score
        ? `${score[1]} - ${score[0]}`
        : reverse.predictedScore,
      predHomeGoals: reverse.predAwayGoals,
      predAwayGoals: reverse.predHomeGoals,
      homeWinProb: reverse.awayWinProb,
      awayWinProb: reverse.homeWinProb,
      predictedOutcome:
        reverse.predictedOutcome === "Home Win"
          ? "Away Win"
          : reverse.predictedOutcome === "Away Win"
            ? "Home Win"
            : reverse.predictedOutcome,
    };
  }, [awayTeam, dataset.fixtures, homeTeam]);
  const scenario = useMemo(
    () => calculateScenario(home, away, homeBoost, awayBoost, neutral),
    [away, awayBoost, home, homeBoost, neutral],
  );
  const baseline = useMemo(
    () => calculateScenario(home, away, 0, 0, false),
    [away, home],
  );
  const resetScenario = () => {
    setHomeTeam(initialHome);
    setAwayTeam(initialAway);
    setHomeBoost(0);
    setAwayBoost(0);
    setNeutral(false);
  };
  if (!home || !away || !scenario)
    return <div className="empty-state">Club data is unavailable.</div>;
  const scenarioFixture = {
    homeWinProb: scenario.homeProb,
    drawProb: scenario.drawProb,
    awayWinProb: scenario.awayProb,
  } as Fixture;
  return (
    <MotionSection className="page-stack">
      <div className="page-intro">
        <div>
          <p className="eyebrow">Scenario tool</p>
          <h1>Ask a different question.</h1>
          <p className="lede">
            Adjust form and venue assumptions to see how the matchup moves.
            Scenario numbers are browser estimates; the scheduled forecast
            remains the production reference.
          </p>
        </div>
      </div>
      <div className="simulator-layout">
        <div className="control-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Adjust assumptions</p>
              <h2>Scenario controls</h2>
            </div>
            <button className="text-button" onClick={resetScenario}>Reset</button>
          </div>
          <label>
            Home club
            <select
              value={homeTeam}
              onChange={(event) => setHomeTeam(event.target.value)}
            >
              {names.map((name) => (
                <option key={name} disabled={name === awayTeam}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Away club
            <select
              value={awayTeam}
              onChange={(event) => setAwayTeam(event.target.value)}
            >
              {names.map((name) => (
                <option key={name} disabled={name === homeTeam}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Home form{" "}
            <input
              type="range"
              min="-30"
              max="30"
              value={homeBoost}
              onChange={(event) => setHomeBoost(Number(event.target.value))}
            />
            <span>
              {homeBoost > 0 ? "+" : ""}
              {homeBoost}%
            </span>
          </label>
          <label>
            Away form{" "}
            <input
              type="range"
              min="-30"
              max="30"
              value={awayBoost}
              onChange={(event) => setAwayBoost(Number(event.target.value))}
            />
            <span>
              {awayBoost > 0 ? "+" : ""}
              {awayBoost}%
            </span>
          </label>
          <label className="check-row">
            <input
              type="checkbox"
              checked={neutral}
              onChange={(event) => setNeutral(event.target.checked)}
            />{" "}
            Neutral venue
          </label>
        </div>
        <div className="scenario-panel">
          <div className="scenario-score">
            <div>
              <TeamMark team={home} />
              <strong>{homeTeam}</strong>
            </div>
            <span>
              {scenario.homeScore} — {scenario.awayScore}
            </span>
            <div>
              <TeamMark team={away} />
              <strong>{awayTeam}</strong>
            </div>
          </div>
          <ProbabilityStrip fixture={scenarioFixture} />
          <div className="probability-labels">
            <span>
              <b>{pct(scenario.homeProb)}</b> Home
            </span>
            <span>
              <b>{pct(scenario.drawProb)}</b> Draw
            </span>
            <span>
              <b>{pct(scenario.awayProb)}</b> Away
            </span>
          </div>
          {baseline && (
            <div className="scenario-delta" aria-live="polite">
              <div>
                <span>Against baseline</span>
                <strong>{scenario.homeProb >= baseline.homeProb ? "Home" : "Away"} moves {deltaLabel(Math.abs(scenario.homeProb - baseline.homeProb))}</strong>
              </div>
              <p>{homeBoost || awayBoost || neutral ? "Your assumptions shift the browser scenario; the scheduled forecast remains unchanged." : "Move a control to compare your scenario with the neutral baseline."}</p>
            </div>
          )}
          <div className="scenario-grid">
            <div>
              <span>Expected home goals</span>
              <strong>{scenario.homeExpected.toFixed(2)}</strong>
            </div>
            <div>
              <span>Expected away goals</span>
              <strong>{scenario.awayExpected.toFixed(2)}</strong>
            </div>
            <div>
              <span>Scheduled forecast</span>
              <strong>{production?.predictedScore ?? "Not scheduled"}</strong>
            </div>
            <div>
              <span>Forecast source</span>
              <strong>{dataset.benchmark.productionModel}</strong>
            </div>
          </div>
        </div>
      </div>
    </MotionSection>
  );
};

const StandingsPage: React.FC<{
  dataset: EPLDataset;
  onClub: (team: string) => void;
}> = ({ dataset, onClub }) => {
  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [ascending, setAscending] = useState(true);
  const sorted = useMemo(
    () =>
      [...dataset.standings].sort((a, b) => {
        const av = a[sortKey];
        const bv = b[sortKey];
        const comparison =
          typeof av === "string"
            ? String(av).localeCompare(String(bv))
            : Number(av) - Number(bv);
        return ascending ? comparison : -comparison;
      }),
    [ascending, dataset.standings, sortKey],
  );
  const chooseSort = (key: SortKey) => {
    if (sortKey === key) setAscending((value) => !value);
    else {
      setSortKey(key);
      setAscending(key === "rank");
    }
  };
  const headers: Array<[SortKey, string]> = [
    ["rank", "#"],
    ["team", "Club"],
    ["played", "P"],
    ["won", "W"],
    ["drawn", "D"],
    ["lost", "L"],
    ["gd", "GD"],
    ["points", "Pts"],
  ];
  return (
    <MotionSection className="page-stack">
      <div className="page-intro">
        <div>
          <p className="eyebrow">Projected league table</p>
          <h1>See the full season projection.</h1>
          <p className="lede">
            Every row is a full-season projection built from the model-s
            scorelines. Browse Fixtures for official results and individual
            forecast reviews.
          </p>
        </div>
      </div>
      <div className="table-note">
        <span className="dot dot-amber" /> Full-season projection{" "}
        <span className="muted">
          Sorted by {sortKey === "rank" ? "table position" : sortKey}
        </span>
      </div>
      <div className="table-wrap">
        <table>
          <caption className="sr-only">Premier League standings</caption>
          <thead>
            <tr>
              {headers.map(([key, label]) => (
                <th
                  key={key}
                  aria-sort={
                    sortKey === key
                      ? ascending
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                >
                  <button onClick={() => chooseSort(key)}>
                    {label}
                    <span aria-hidden="true">
                      {sortKey === key ? (ascending ? " ↑" : " ↓") : ""}
                    </span>
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((row) => (
              <tr
                key={row.team}
                onClick={() => onClub(row.team)}
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onClub(row.team);
                  }
                }}
                aria-label={`Open ${row.team} club profile`}
                title={`Open ${row.team} club profile`}
              >
                <td>{row.rank}</td>
                <td>
                  <span className="club-cell">
                    <TeamMark
                      team={dataset.teams[row.team]}
                      short={row.short}
                    />
                    <strong>{row.team}</strong>
                  </span>
                </td>
                <td>{row.played}</td>
                <td>{row.won}</td>
                <td>{row.drawn}</td>
                <td>{row.lost}</td>
                <td>{row.gd > 0 ? `+${row.gd}` : row.gd}</td>
                <td>
                  <strong>{row.points}</strong>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </MotionSection>
  );
};

const ClubsPage: React.FC<{
  dataset: EPLDataset;
  selectedClub: string;
  setSelectedClub: (club: string) => void;
  onSimulate: (home: string, away: string) => void;
}> = ({ dataset, selectedClub, setSelectedClub, onSimulate }) => {
  const names = useMemo(
    () =>
      Object.values(dataset.teams)
        .sort((a, b) => a.rank - b.rank)
        .map((team) => team.name),
    [dataset.teams],
  );
  const profile = dataset.teams[selectedClub] ?? dataset.teams[names[0]];
  if (!profile)
    return <div className="empty-state">No club data available.</div>;
  const clubFixtures = dataset.fixtures
    .filter(
      (fixture) =>
        fixture.homeTeam === profile.name || fixture.awayTeam === profile.name,
    )
    .sort((a, b) => a.gameweek - b.gameweek);
  const upcoming = clubFixtures
    .filter((fixture) => fixture.status !== "Played")
    .slice(0, 5);
  const recent = clubFixtures
    .filter((fixture) => fixture.status === "Played")
    .slice(-5)
    .reverse();
  return (
    <MotionSection className="page-stack clubs-page">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Club profile</p>
          <h1>Follow one club through the season.</h1>
        </div>
        <select
          className="compact-select"
          value={profile.name}
          onChange={(event) => setSelectedClub(event.target.value)}
          aria-label="Select a club"
        >
          {names.map((name) => (
            <option key={name}>{name}</option>
          ))}
        </select>
      </div>
      <article className="club-hero" style={{ borderTopColor: profile.color }}>
        <div className="club-identity">
          <TeamMark team={profile} />
          <div>
            <p className="eyebrow">
              #{profile.rank} · {profile.points} points
            </p>
            <h2>{profile.name}</h2>
            <p className="muted">{profile.stadium}</p>
          </div>
        </div>
        <div className="club-kpis">
          <div>
            <span>Record</span>
            <strong>
              {profile.won}–{profile.drawn}–{profile.lost}
            </strong>
          </div>
          <div>
            <span>Goals</span>
            <strong>
              {profile.gf}–{profile.ga}
            </strong>
          </div>
          <div>
            <span>Win rate</span>
            <strong>{profile.winRate}%</strong>
          </div>
          <div>
            <span>Avg rest</span>
            <strong>{profile.restDaysAvg}d</strong>
          </div>
        </div>
      </article>
      <div className="two-column">
        <div className="content-card">
          <div className="section-heading small">
            <h2>Recent results</h2>
            <span className="muted">Official</span>
          </div>
          {recent.length ? (
            recent.map((fixture) => (
              <div className="mini-fixture" key={fixture.id}>
                <span
                  className={`result-badge ${resultTone(clubResultFor(fixture, profile.name))}`}
                >
                  {clubResultFor(fixture, profile.name)?.[0] ?? "—"}
                </span>
                <span>
                  {fixture.homeTeam === profile.name ? "vs" : "@"}{" "}
                  {fixture.homeTeam === profile.name
                    ? fixture.awayTeam
                    : fixture.homeTeam}
                </span>
                <strong>{fixture.actualScore}</strong>
              </div>
            ))
          ) : (
            <p className="muted">No completed fixtures.</p>
          )}
        </div>
        <div className="content-card">
          <div className="section-heading small">
            <h2>Next fixtures</h2>
            <span className="muted">Model score</span>
          </div>
          {upcoming.length ? (
            upcoming.map((fixture) => (
              <button
                className="mini-fixture interactive"
                key={fixture.id}
                onClick={() => onSimulate(fixture.homeTeam, fixture.awayTeam)}
              >
                <span className="muted">GW{fixture.gameweek}</span>
                <span>
                  {fixture.homeTeam === profile.name ? "vs" : "@"}{" "}
                  {fixture.homeTeam === profile.name
                    ? fixture.awayTeam
                    : fixture.homeTeam}
                </span>
                <strong>{fixture.predictedScore}</strong>
              </button>
            ))
          ) : (
            <p className="muted">Season complete.</p>
          )}
        </div>
      </div>
    </MotionSection>
  );
};

const AnalyticsPage: React.FC<{ dataset: EPLDataset }> = ({ dataset }) => {
  const [selectedImage, setSelectedImage] = useState<{
    src: string;
    title: string;
    caption: string;
  } | null>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!selectedImage) return;
    const opener = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSelectedImage(null);
        return;
      }
      if (event.key === "Tab" && dialogRef.current) {
        const focusables = Array.from(
          dialogRef.current.querySelectorAll<HTMLElement>(
            'button, [href], img[tabindex], [tabindex]:not([tabindex="-1"])',
          ),
        ).filter((el) => !el.hasAttribute("disabled"));
        if (!focusables.length) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first.focus();
        }
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => {
      window.removeEventListener("keydown", handleKey);
      opener?.focus?.();
    };
  }, [selectedImage]);
  const model =
    dataset.benchmark.models.find((entry) => entry.isProduction) ??
    dataset.benchmark.models[0];
  const asset = (path: string) =>
    `${import.meta.env.BASE_URL}${path.replace(/^\.?\//, "")}`;
  const modelChartData = dataset.benchmark.models.map((entry) => ({
    name: entry.name.replace("Random Forest", "RF").replace("Logistic Regression", "Logistic"),
    rps: entry.rps,
    isProduction: entry.isProduction,
  }));
  const rpsMin = Math.min(...modelChartData.map((entry) => entry.rps));
  const rpsMax = Math.max(...modelChartData.map((entry) => entry.rps));
  const rpsRange = Math.max(rpsMax - rpsMin, 0.001);
  const playedFixtures = dataset.fixtures.filter((fixture) => fixture.status === "Played").length;
  const projectedFixtures = dataset.fixtures.length - playedFixtures;
  const coveragePercent = dataset.fixtures.length > 0
    ? Math.round((playedFixtures / dataset.fixtures.length) * 100)
    : 0;
  const firstKickoffByGameweek = new Map<number, string>();
  for (const fixture of dataset.fixtures) {
    if (fixture.date !== "TBC" && !firstKickoffByGameweek.has(fixture.gameweek)) {
      firstKickoffByGameweek.set(fixture.gameweek, fixture.date);
    }
  }
  const goalChartData = dataset.analytics.goalsPerGameweek.map((entry) => ({
    date: new Date(
      firstKickoffByGameweek.get(entry.gw) ??
        Date.UTC(2026, 7, 14 + (entry.gw - 1) * 7),
    ),
    goals: entry.goals,
    average: entry.avgPerMatch,
    gameweek: entry.gw,
  }));
  const outcome = dataset.analytics.outcomeDistribution;
  const outcomeChartData = [
    { label: "Home", value: outcome.homePct, maxValue: 100, color: "#1d6f52" },
    { label: "Draw", value: outcome.drawPct, maxValue: 100, color: "#b46b2a" },
    { label: "Away", value: outcome.awayPct, maxValue: 100, color: "#3d5a80" },
  ];
  return (
    <MotionSection className="page-stack">
      <div className="page-intro">
        <div>
          <p className="eyebrow">Analytics</p>
          <h1>Read the season through its evidence.</h1>
          <p className="lede">
            Metrics are calculated on a time ordered holdout. This page
            separates validation evidence from the season projection so the
            interface never presents a scenario as a measured result.
          </p>
        </div>
        <div className="intro-stat">
          <strong>{dataset.benchmark.productionModel}</strong>
          <span>selected for production</span>
        </div>
      </div>
      <div className="metric-grid">
        <div className="metric-primary">
          <span>RPS</span>
          <strong>{model?.rps ?? "—"}</strong>
          <small>Primary selection metric · lower is better</small>
        </div>
        <div>
          <span>Accuracy</span>
          <strong>{model?.accuracy ?? "—"}%</strong>
          <small>Outcome argmax</small>
        </div>
        <div>
          <span>Log loss</span>
          <strong>{model?.logLoss ?? "—"}</strong>
          <small>Probability quality</small>
        </div>
        <div>
          <span>Goal MAE</span>
          <strong>{model?.avgGoalMae ?? "—"}</strong>
          <small>Expected goals</small>
        </div>
        <div>
          <span>Within one goal</span>
          <strong>{model?.within1Goal ?? "—"}%</strong>
          <small>Scoreline tolerance</small>
        </div>
      </div>
      <div className="analytics-chart-grid" aria-label="Interactive analytics charts">
        <Card className="analytics-chart-card analytics-chart-wide">
          <CardHeader>
            <div className="analytics-card-heading">
              <div>
                <CardTitle>RPS benchmark</CardTitle>
                <CardDescription>Lower scores indicate better-calibrated probabilities.</CardDescription>
              </div>
              <Badge variant="outline">Production: {dataset.benchmark.productionModel}</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div
              className="rps-chart"
              role="img"
              aria-label="RPS comparison. Lower scores are better."
            >
              <div className="rps-chart-scale" aria-hidden="true">
                <span>Higher bars = higher RPS</span>
                <span>Zoomed to observed range</span>
                <span>Lower score = better</span>
              </div>
              <div className="rps-bars">
                {modelChartData.map((entry) => {
                  const relativeHeight = 34 + ((entry.rps - rpsMin) / rpsRange) * 66;
                  return (
                    <div className={`rps-bar-group ${entry.isProduction ? "is-production" : ""}`} key={entry.name}>
                      <strong>{entry.rps.toFixed(4)}</strong>
                      <div className="rps-bar-track">
                        <div className="rps-bar" style={{ height: `${relativeHeight}%` }} />
                      </div>
                      <span>{entry.name}</span>
                      {entry.isProduction && <small>Production</small>}
                    </div>
                  );
                })}
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="analytics-chart-card">
          <CardHeader>
            <CardTitle>Outcome mix</CardTitle>
            <CardDescription>Share across official and projected fixtures.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="outcome-chart-wrap">
              <RingChart data={outcomeChartData} strokeWidth={15} ringGap={8} baseInnerRadius={38}>
                <Ring index={0} color="#1d6f52" />
                <Ring index={1} color="#b46b2a" />
                <Ring index={2} color="#3d5a80" />
              </RingChart>
              <div className="outcome-chart-total"><strong>{dataset.fixtures.length}</strong><span>fixtures</span></div>
            </div>
            <div className="chart-legend">
              {outcomeChartData.map((entry) => <span key={entry.label}><i style={{ background: entry.color }} />{entry.label} {entry.value.toFixed(1)}%</span>)}
            </div>
          </CardContent>
        </Card>
        <Card className="analytics-chart-card coverage-card">
          <CardHeader>
            <CardTitle>Season coverage</CardTitle>
            <CardDescription>How much of the schedule has a recorded result.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="coverage-stat">
              <strong>{playedFixtures}<span>/{dataset.fixtures.length}</span></strong>
              <div>
                <span>results recorded</span>
                <small>{coveragePercent}% of the season</small>
              </div>
            </div>
            <div className="coverage-meter" aria-label={`${coveragePercent}% of fixtures have recorded results`} role="progressbar" aria-valuemax={100} aria-valuemin={0} aria-valuenow={coveragePercent}>
              <span style={{ width: `${coveragePercent}%` }} />
            </div>
            <div className="coverage-breakdown">
              <span><i className="coverage-dot recorded" />{playedFixtures} played</span>
              <span><i className="coverage-dot projected" />{projectedFixtures} projected</span>
            </div>
          </CardContent>
        </Card>
        <Card className="analytics-chart-card analytics-chart-wide">
          <CardHeader>
            <div className="analytics-card-heading">
              <div>
                <CardTitle>Goals by gameweek</CardTitle>
                <CardDescription>Observed and projected goal totals across the season schedule.</CardDescription>
              </div>
              <Badge variant="secondary">Offline dataset</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="bklit-chart-shell">
              <LineChart data={goalChartData} xDataKey="date" xPadding={10} aspectRatio="2.25 / 1" margin={{ top: 20, right: 16, bottom: 42, left: 16 }}>
                <Grid horizontal stroke="rgba(23,60,50,.12)" strokeDasharray="2,4" />
                <Line dataKey="goals" stroke="#1d6f52" strokeWidth={3} showMarkers />
                <XAxis numTicks={7} />
                <ChartTooltip rows={(point) => [{ label: `GW${point.gameweek}`, value: `${point.goals} goals`, color: "#1d6f52" }]} />
              </LineChart>
            </div>
          </CardContent>
        </Card>
      </div>
      <div className="content-card">
        <div className="section-heading small">
          <div>
            <p className="eyebrow">Validation</p>
            <h2>Model comparison</h2>
          </div>
          <span className="muted">
            Lower log loss and goal error are better
          </span>
        </div>
        <div className="model-table">
          <div className="model-row model-header">
            <span>Model</span>
            <span>RPS</span>
            <span>Accuracy</span>
            <span>Log loss</span>
            <span>Goal MAE</span>
          </div>
          {dataset.benchmark.models.map((entry) => (
            <div
              className={`model-row ${entry.isProduction ? "selected-row" : ""}`}
              key={entry.name}
            >
              <strong>
                {entry.name}
                {entry.isProduction && <em>Production</em>}
              </strong>
              <span data-label="RPS" aria-label={`RPS ${entry.rps}`}>{entry.rps}</span>
              <span data-label="Accuracy" aria-label={`Accuracy ${entry.accuracy}%`}>{entry.accuracy}%</span>
              <span data-label="Log loss" aria-label={`Log loss ${entry.logLoss}`}>{entry.logLoss}</span>
              <span data-label="Goal MAE" aria-label={`Goal MAE ${entry.avgGoalMae}`}>{entry.avgGoalMae}</span>
            </div>
          ))}
        </div>
        <div className="metric-explainer">
          <div>
            <p className="eyebrow">Why RPS leads</p>
            <h3>Probability quality matters more than a single winner.</h3>
          </div>
          <p>Ranked Probability Score rewards calibrated probability distributions, not just the most likely outcome. The production model is selected by the lowest evaluation RPS, with log loss, accuracy, and goal error retained as supporting evidence.</p>
        </div>
      </div>
      <div className="content-card">
        <div className="section-heading small">
          <div>
            <p className="eyebrow">Diagnostics</p>
            <h2>Training signals</h2>
          </div>
          <span className="muted">Generated from the current benchmark</span>
        </div>
        <div className="diagnostic-grid">
          {dataset.benchmark.diagnostics.map((diagnostic) => (
            <button
              key={diagnostic.id}
              onClick={() => setSelectedImage(diagnostic)}
            >
              <img
                src={asset(diagnostic.src)}
                alt={diagnostic.title}
                loading="lazy"
              />
              <span>
                <strong>{diagnostic.title}</strong>
                <small>{diagnostic.caption}</small>
              </span>
            </button>
          ))}
        </div>
      </div>
      {selectedImage && (
        <div
          className="dialog-backdrop"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setSelectedImage(null);
          }}
        >
          <div
            className="image-dialog"
            ref={dialogRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="diagnostic-title"
          >
            <div className="section-heading small">
              <h2 id="diagnostic-title">{selectedImage.title}</h2>
              <button
                ref={closeRef}
                onClick={() => setSelectedImage(null)}
                aria-label="Close diagnostic preview"
              >
                <X size={18} />
              </button>
            </div>
            <img src={asset(selectedImage.src)} alt={selectedImage.title} />
            <p className="muted">{selectedImage.caption}</p>
          </div>
        </div>
      )}
    </MotionSection>
  );
};

const Dashboard: React.FC<{
  dataset: EPLDataset;
  initialTab: Tab;
  onNavigate: (route: AppRoute) => void;
}> = ({ dataset, initialTab, onNavigate }) => {
  const [activeTab, setActiveTab] = useState<Tab>(initialTab);
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [selectedClub, setSelectedClub] = useState("Arsenal");
  const [motionDisabled, setMotionDisabled] = useState(false);
  const [simulatorSelection, setSimulatorSelection] = useState({
    home: "Arsenal",
    away: "Chelsea",
  });
  useEffect(() => setActiveTab(initialTab), [initialTab]);
  const navigateTo = (tab: Tab) => {
    setActiveTab(tab);
    onNavigate(tab);
  };
  const searchRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onPointer = (event: PointerEvent) => {
      if (!searchRef.current?.contains(event.target as Node))
        setSearchOpen(false);
    };
    document.addEventListener("pointerdown", onPointer);
    return () => document.removeEventListener("pointerdown", onPointer);
  }, []);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "/" && document.activeElement?.tagName !== "INPUT") {
        event.preventDefault();
        (
          searchRef.current?.querySelector("input") as HTMLInputElement | null
        )?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  const selectClub = (club: string) => {
    setSelectedClub(club);
    setQuery("");
    setSearchOpen(false);
    navigateTo("clubs");
  };
  const goHome = () => {
    setQuery("");
    setSearchOpen(false);
    navigateTo("fixtures");
  };
  const openSimulator = (home: string, away: string) => {
    setSimulatorSelection({ home, away });
    setQuery("");
    setSearchOpen(false);
    navigateTo("simulator");
  };
  const clubHits = query
    ? Object.values(dataset.teams)
        .filter((team) => team.name.toLowerCase().includes(query.toLowerCase()))
        .slice(0, 4)
    : [];
  const fixtureHits = query
    ? dataset.fixtures
        .filter((fixture) =>
          `${fixture.homeTeam} ${fixture.awayTeam}`
            .toLowerCase()
            .includes(query.toLowerCase()),
        )
        .slice(0, 4)
    : [];
  const active = tabs.find((tab) => tab.id === activeTab) ?? tabs[0];
  return (
    <MotionPreferenceProvider disabled={motionDisabled}>
      <div className="app-shell">
      <aside className="site-rail">
        <button
          type="button"
          className="brand brand-home"
          onClick={goHome}
          aria-label="Back to fixtures home"
        >
          <img
            className="brand-logo"
            src={`${import.meta.env.BASE_URL}epl-predictor-header-logo.png`}
            alt="EPL Predictor"
            draggable={false}
          />
          <div className="brand-copy">
            <small>Match analysis · {dataset.season}</small>
          </div>
        </button>
        <nav aria-label="Primary navigation">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={activeTab === tab.id ? "active" : ""}
              onClick={() => navigateTo(tab.id)}
              aria-current={activeTab === tab.id ? "page" : undefined}
            >
              <span>{tab.label}</span>
              <small>{tab.note}</small>
            </button>
          ))}
        </nav>
        <div className="rail-footer">
          <span>Data snapshot</span>
          <strong>{dataset.totalMatches} fixtures</strong>
          <small>Offline dataset · no live API</small>
          <small>Scores and schedules are versioned with the export.</small>
        </div>
      </aside>
      <div className="site-content">
        <header className="site-header">
          <div>
            <p className="eyebrow">Premier League · {dataset.season}</p>
            <h2>{active.label}</h2>
          </div>
          <div className="search-box" ref={searchRef}>
            <Search size={17} aria-hidden="true" />
            <input
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setSearchOpen(true);
              }}
              onFocus={() => setSearchOpen(true)}
              placeholder="Search clubs or fixtures"
              aria-label="Search clubs or fixtures"
            />
            <kbd>/</kbd>
            {searchOpen && query && (
              <div className="search-results" role="listbox">
                {clubHits.map((team) => (
                  <button
                    key={team.name}
                    onClick={() => selectClub(team.name)}
                    role="option"
                  >
                    <TeamMark team={team} />
                    {team.name}
                    <small>Club profile</small>
                  </button>
                ))}
                {fixtureHits.map((fixture) => (
                  <button
                    key={fixture.id}
                    onClick={() =>
                      openSimulator(fixture.homeTeam, fixture.awayTeam)
                    }
                    role="option"
                  >
                    <span>
                      {fixture.homeTeam} v {fixture.awayTeam}
                    </span>
                    <small>Open simulator</small>
                  </button>
                ))}
                {!clubHits.length && !fixtureHits.length && (
                  <p className="muted">No matches found.</p>
                )}
              </div>
            )}
          </div>
        </header>
        <main>
          <MotionSection key={activeTab} className="page-transition-shell">
          {activeTab === "fixtures" && (
            <FixturesPage
              key="fixtures"
              dataset={dataset}
              query={query}
              onSimulate={openSimulator}
            />
          )}
          {activeTab === "simulator" && (
            <SimulatorPage
              key="simulator"
              dataset={dataset}
              initialHome={simulatorSelection.home}
              initialAway={simulatorSelection.away}
            />
          )}
          {activeTab === "standings" && (
            <StandingsPage key="standings" dataset={dataset} onClub={selectClub} />
          )}
          {activeTab === "clubs" && (
            <ClubsPage
              key="clubs"
              dataset={dataset}
              selectedClub={selectedClub}
              setSelectedClub={setSelectedClub}
              onSimulate={openSimulator}
            />
          )}
          {activeTab === "analytics" && <AnalyticsPage key="analytics" dataset={dataset} />}
          </MotionSection>
        </main>
        <footer className="site-footer">
          <span>Forecasts are probabilities, not guarantees.</span>
          <button className="text-button motion-toggle" onClick={() => setMotionDisabled((value) => !value)} aria-pressed={motionDisabled}>
            {motionDisabled ? "Motion reduced" : "Motion on"}
          </button>
          <a
            href="https://github.com/Raghav2012Code/epl-predictor"
            target="_blank"
            rel="noreferrer"
          >
            View source <ArrowUpRight size={14} />
          </a>
        </footer>
      </div>
      </div>
    </MotionPreferenceProvider>
  );
};

export const App: React.FC = () => {
  const state = useEPLData();
  const [route, setRoute] = useState<AppRoute>(() =>
    appRouteForPath(window.location.pathname),
  );
  useEffect(() => {
    const onPopState = () => setRoute(appRouteForPath(window.location.pathname));
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);
  const navigate = (nextRoute: AppRoute) => {
    const nextPath = pathForAppRoute(nextRoute);
    if (window.location.pathname !== nextPath) {
      window.history.pushState({}, "", nextPath);
    }
    setRoute(nextRoute);
    window.scrollTo({ top: 0, behavior: "auto" });
  };
  if (state.status === "loading")
    return (
      <ErrorBoundary>
        {route === "landing" ? (
          <LandingPage onNavigate={navigate} />
        ) : (
          <AppSkeleton />
        )}
      </ErrorBoundary>
    );
  if (state.status === "error")
    return (
      <ErrorBoundary>
        <DataErrorPanel error={state.error} onRetry={state.retry} />
      </ErrorBoundary>
    );
  return (
    <ErrorBoundary>
      {route === "landing" ? (
        <LandingPage dataset={state.dataset} onNavigate={navigate} />
      ) : (
        <Dashboard
          dataset={state.dataset}
          initialTab={route}
          onNavigate={navigate}
        />
      )}
    </ErrorBoundary>
  );
};
export default App;
