// Serves the engine and the networks' weights from the R2 bucket (binding BUCKET, wrangler.toml)
// at /assets/<name>. Their names hold a hash of their content (web/build_site.py), so a name
// never changes content and browsers may keep them for a year.
async function serve({ params, env, request }) {
  const key = (params.path || []).join("/");
  const object = await env.BUCKET.get(key, { onlyIf: request.headers });
  if (object === null) return new Response("Not found", { status: 404 });
  const headers = new Headers();
  object.writeHttpMetadata(headers); // the content type set on upload (web/deploy.sh)
  headers.set("ETag", object.httpEtag);
  headers.set("Cache-Control", "public, max-age=31536000, immutable");
  headers.set("X-Content-Type-Options", "nosniff");
  // A conditional request whose object is unchanged comes back without a body.
  if (!object.body) return new Response(null, { status: 304, headers });
  return new Response(object.body, { headers });
}

export const onRequestGet = serve;

// HEAD gets the same headers without a body (otherwise it falls through to the page).
export async function onRequestHead(context) {
  const r = await serve(context);
  return new Response(null, { status: r.status, headers: r.headers });
}
