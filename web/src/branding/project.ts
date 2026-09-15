export interface ProjectMember {
  readonly name: string;
  readonly role?: string;
  readonly orcid?: string;
}

export const PROJECT_TEAM: readonly ProjectMember[] = [
  {
    name: "Alin Marin Elena",
    role: "Principal Investigator",
    orcid: "0000-0002-7013-6670",
  },
  {
    name: "Susmita Basak",
    role: "Co-Investigator",
    orcid: "0000-0003-3122-0308",
  },
  { name: "Junwen Yin", orcid: "0000-0001-7374-9352" },
  { name: "Gilberto Teobaldi", orcid: "0000-0001-6068-6786" },
  { name: "Jaehoon Cha", orcid: "0000-0002-2498-4214" },
  { name: "Willow Sparks", orcid: "0009-0008-7630-9609" },
];

export const FUNDING_GRANT = "EP/Z530657/1";
export const FUNDING_URL = "https://gtr.ukri.org/projects?ref=EP%2FZ530657%2F1";
export const PROJECT_URL = "https://goldilocks.ac.uk/";
export const PROJECT_DESCRIPTION =
  "Convergence tools and evidence-based best practices for numerical approximations in Density Functional Theory (DFT) calculations.";
