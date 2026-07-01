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

  // Derive the subpath directly from the raw request URL rather than
  // req.query.path — the dynamic-route query param is not reliably populated
  // for this catch-all function (observed empty in production, which silently
  // forwarded every call to "<vpsBase>/api/" with no subpath, hitting no route
  // on the VPS and returning a generic 404 instead of real data).
  const requestUrl = new URL(req.url, "http://localhost");
  const subpath = requestUrl.pathname.replace(/^\/api\/?/, "");

  const qsStr = requestUrl.searchParams.toString();
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
