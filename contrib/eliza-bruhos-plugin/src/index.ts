/**
 * bruhOS skills plugin for elizaOS / odiza.
 *
 * Handlers never touch ActionGroupControl. They POST to the Pi skill API and
 * return ActionResult { success, text, data } from the Outcome.
 *
 * Catalog can be refreshed from GET /v1/skills (optional) or hard-coded from
 * the shipped list below — keep in sync with bruhos.skills.builtin.
 */

export type ActionResult = {
  success: boolean;
  text?: string;
  data?: Record<string, unknown>;
  error?: string;
};

type SkillArg = {
  type: string;
  required?: boolean;
  choices?: string[];
  default?: unknown;
  doc?: string;
};

type SkillDesc = {
  skill: string;
  doc: string;
  args: Record<string, SkillArg>;
};

/** Minimal subset matching bruhOS go + common motion skills. Extend as needed. */
export const SKILLS: SkillDesc[] = [
  {
    skill: "go",
    doc: 'Open-space gait. Use for "walk forward" / turn left|right. No vision target. steps capped at 6.',
    args: {
      direction: { type: "string", required: true, choices: ["forward", "left", "right"] },
      steps: { type: "integer", default: 3 },
    },
  },
  {
    skill: "wave",
    doc: "Wave hello.",
    args: {},
  },
  {
    skill: "stop",
    doc: "Halt all motion now.",
    args: {},
  },
  {
    skill: "kick",
    doc: "Kick with the given foot.",
    args: {
      foot: { type: "string", default: "right", choices: ["left", "right"] },
    },
  },
  {
    skill: "walk_toward",
    doc: "Walk until a seen object is within reach. Needs a visual target. Prefer go when there is no object.",
    args: {
      target: { type: "string", required: true },
      stop_distance_m: { type: "number", default: 0.1 },
    },
  },
];

const skillsUrl =
  process.env.BRUHOS_SKILLS_URL || "http://127.0.0.1:8090/v1/skills/run";

async function runSkill(
  skill: string,
  args: Record<string, unknown>,
): Promise<ActionResult> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (process.env.BRUHOS_SKILLS_TOKEN) {
    headers.Authorization = `Bearer ${process.env.BRUHOS_SKILLS_TOKEN}`;
  }
  const res = await fetch(skillsUrl, {
    method: "POST",
    headers,
    body: JSON.stringify({ skill, args }),
  });
  const body = (await res.json()) as {
    ok?: boolean;
    status?: string;
    detail?: string;
    data?: Record<string, unknown>;
    error?: string;
  };
  if (!res.ok) {
    return {
      success: false,
      text: body.error || body.detail || `skill HTTP ${res.status}`,
      error: body.error || body.detail,
    };
  }
  return {
    success: Boolean(body.ok) || body.status === "ok" || body.status === "stopped",
    text: body.detail || body.status || "done",
    data: body.data,
  };
}

function actionFor(desc: SkillDesc) {
  const name = desc.skill.toUpperCase();
  return {
    name,
    similes: [desc.skill.replace(/_/g, " "), desc.skill],
    description: desc.doc,
    validate: async () => true,
    handler: async (
      _runtime: unknown,
      message: { content?: { text?: string } },
      _state: unknown,
      options?: { parameters?: Record<string, unknown> },
    ): Promise<ActionResult> => {
      const params = options?.parameters || {};
      // Prefer structured params from the planner; fall back to empty.
      return runSkill(desc.skill, params as Record<string, unknown>);
    },
  };
}

/**
 * Plugin export. Load from odiza or eliza character config:
 *   plugins: ["@bruhos/plugin-skills"]
 *
 * Also register an HTTP route POST /bruhos/message that runs the agent turn
 * and returns { text } — wire that in your host (path differs per eliza version).
 */
export const bruhosPlugin = {
  name: "@bruhos/plugin-skills",
  description: "TonyPi skills via bruhOS Pi skill API",
  actions: SKILLS.map(actionFor),
};

export default bruhosPlugin;
