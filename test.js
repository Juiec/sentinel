// Test: start the exported app on a random port, send GET /, assert 200.
const net = require('node:net');
const assert = require('node:assert');
const app = require('./server.js');

function getFreePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.on('error', reject);
    srv.listen(0, () => {
      const port = srv.address().port;
      srv.close(() => resolve(port));
    });
  });
}

async function main() {
  const port = await getFreePort();
  const server = app.listen(port);
  // Wait for the actual 'listening' event instead of a fixed timeout.
  await new Promise((resolve, reject) => {
    server.once('listening', resolve);
    server.once('error', reject);
  });

  const res = await fetch(`http://localhost:${port}/`);
  assert.strictEqual(res.status, 200);
  await res.body.cancel();
  server.close();
  console.log('test passed: GET / returned 200');
}

main().catch((e) => {
  console.error(e.message || e);
  process.exitCode = 1;
});
