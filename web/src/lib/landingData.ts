import type { EPLDataset, Fixture } from "../types";

export type LandingData = {
  season: string;
  totalMatches: number;
  playedMatches: number;
  fixture: Fixture | undefined;
  productionModel: string;
  productionRps: number;
};

export const getLandingData = (dataset?: EPLDataset | null): LandingData => {
  const productionModel = dataset?.benchmark.productionModel ?? "Stacked";
  const productionRps = dataset?.benchmark.models.find(
    (model) => model.name === productionModel,
  )?.rps ?? 0.2044;

  return {
    season: dataset?.season ?? "2026/2027",
    totalMatches: dataset?.totalMatches ?? 380,
    playedMatches: dataset?.fixtures.filter((fixture) => fixture.status === "Played").length ?? 0,
    fixture: dataset?.fixtures.find((fixture) => fixture.status !== "Played") ?? dataset?.fixtures[0],
    productionModel,
    productionRps,
  };
};
