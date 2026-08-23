/** A failed API response, with the server's `detail` preserved.
 *
 * WHY: `apiFetch` used to throw `new Error("503 Service Unavailable")` and
 * discard the body. The platform now refuses to publish any worker command
 * unless `TREEMINER_PLATFORM_COMMAND_SECRET` is configured, and it says so in
 * the 503's `detail`. Throwing away that body turned an operator-fixable
 * misconfiguration into a mystery.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/** What the operator has to actually do about it. */
export const COMMAND_SECRET_MISSING_MESSAGE =
  "Commands are disabled: the platform command secret is not configured on " +
  "the server. Set TREEMINER_PLATFORM_COMMAND_SECRET in the server's " +
  "environment and restart it.";

/** Substring of server/command_signing.py's CommandSecretMissing message. */
const SECRET_MISSING_MARKER = "command secret is not configured";

/** True iff this failure is the server refusing to sign, not a transport fault. */
export function isCommandSecretMissing(error: unknown): boolean {
  return (
    error instanceof ApiError &&
    error.status === 503 &&
    error.detail.toLowerCase().includes(SECRET_MISSING_MARKER)
  );
}

/** FastAPI puts the message in `detail`, which is a string or a list of errors. */
function extractDetail(body: string): string {
  try {
    const parsed = JSON.parse(body);
    const detail = parsed?.detail;
    if (typeof detail === "string") return detail;
    if (detail !== undefined) return JSON.stringify(detail);
  } catch {
    /* not JSON: fall through to the raw body */
  }
  return body;
}

export async function apiFetch<T = unknown>(
  url: string,
  init?: RequestInit,
): Promise<T> {
  const headers = new Headers(init?.headers);
  const res = await fetch(url, { ...init, headers, credentials: "same-origin" });
  if (!res.ok) {
    const detail = extractDetail(await res.text().catch(() => ""));
    const error = new ApiError(
      res.status,
      detail,
      detail || `${res.status} ${res.statusText}`,
    );
    // Replace the raw server prose with the actionable version, once, here, so
    // every command site reports it identically without each one knowing about
    // command signing.
    if (isCommandSecretMissing(error)) {
      return Promise.reject(
        new ApiError(res.status, detail, COMMAND_SECRET_MISSING_MESSAGE),
      );
    }
    throw error;
  }
  return res.json();
}
