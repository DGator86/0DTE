/**
 * Vercel serverless proxy — forwards read-only /api/* to the VPS dashboard.
 * Set VPS_API_URL and DASHBOARD_TOKEN in Vercel project environment variables.
 * The browser never sees the token; Vercel adds it server-side.
 */
export default async function handler(req, res) {
  if (!["GET", "HEAD", "OPTIONS"].includes(req.method)) {
    return res.status(405).json({ detail: "Method not allowed — read-only dashboard" });
  }

  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }

  const vpsBase = process.env.VPS_API_URL?.replace(/\/$/, "");
  const token = process.env.DASHBOARD_TOKEN;

  if (!vpsBase || !token) {
    return res.status(503).json({
      detail: "Vercel env not configured — set VPS_API_URL and DASHBOARD_TOKEN",
    });
  }

  const pathParts = req.query.path;
  const subpath = Array.isArray(pathParts) ? pathParts.join("/") : pathParts || "";

  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(req.query)) {
    if (key === "path") continue;
    if (Array.isArray(value)) {
      value.forEach((v) => qs.append(key, v));
    } else if (value != null) {
      qs.append(key, value);
    }
  }
  const qsStr = qs.toString();
  const target = `${vpsBase}/api/${subpath}${qsStr ? `?${qsStr}` : ""}`;

  try {
    const upstream = await fetch(target, {
      method: req.method,
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: req.headers.accept || "application/json",
      },
    });

    const contentType = upstream.headers.get("content-type") || "application/json";
    res.status(upstream.status);
    res.setHeader("Content-Type", contentType);
    res.setHeader("Cache-Control", "no-store");

    const body = await upstream.text();
    return res.send(body);
  } catch (err) {
    return res.status(502).json({
      detail: "VPS dashboard unreachable — check VPS_API_URL and tunnel",
      error: err instanceof Error ? err.message : String(err),
    });
  }
}
