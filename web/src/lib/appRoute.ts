export type AppRoute =
  | "landing"
  | "fixtures"
  | "simulator"
  | "standings"
  | "clubs"
  | "analytics";

const routePaths: Record<AppRoute, string> = {
  landing: "/",
  fixtures: "/fixtures",
  simulator: "/simulator",
  standings: "/table",
  clubs: "/clubs",
  analytics: "/analytics",
};

export const dashboardRoutes: Array<Exclude<AppRoute, "landing">> = [
  "fixtures",
  "simulator",
  "standings",
  "clubs",
  "analytics",
];

const routesByPath = Object.fromEntries(
  Object.entries(routePaths).map(([route, path]) => [path, route]),
) as Record<string, AppRoute>;

export const appRouteForPath = (pathname: string): AppRoute =>
  routesByPath[pathname.replace(/\/$/, "") || "/"] ?? "landing";

export const pathForAppRoute = (route: AppRoute): string => routePaths[route];
