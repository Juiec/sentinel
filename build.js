// build.js — runs before start/deploy. Mirrors what Vercel does for you:
// 1. sanity-check the environment (node version, deps present)
// 2. verify public assets exist
// 3. smoke-test that server.js actually boots and serves index.html

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

console.log('\n🔧 Building sentinel...\n');

// --- 1. Environment check ---------------------------------------------
const major = parseInt(process.version.slice(1), 10);
if (major < 18) {
  console.error(`❌ Node ${process.version} is too old. Install Node 18+ (https://nodejs.org).`);
  process.exit(1);
}
console.log(`✅ Node ${process.version}`);

// --- 2. Assets check ----------------------------------------------------
const publicDir = path.join(__dirname, 'public');
if (!fs.existsSync(publicDir)) {
  console.error('❌ Missing ./public folder — create it and add index.html.');
  process.exit(1);
}
const indexHtml = path.join(publicDir, 'index.html');
if (!fs.existsSync(indexHtml)) {
  console.error('❌ Missing public/index.html');
  process.exit(1);
}
console.log(`✅ public/ assets present (${fs.readdirSync(publicDir).join(', ')})`);

// --- 3. Smoke-test the server ------------------------------------------
const child = spawn(process.execPath, ['server.js'], { env: { ...process.env, PORT: '3999' } });
let checked = false;

setTimeout(() => {
  fetch('http://localhost:3999/')
    .then((res) => res.text())
    .then((html) => {
      checked = true;
      child.kill();
      if (!html.includes('<title>')) {
        console.error('❌ Server responded but did not serve index.html');
        process.exit(1);
      }
      console.log('✅ server.js boots and serves index.html on PORT 3999');
      console.log('\n🏁 Build complete. Run `npm start` to run it.\n');
      process.exit(0);
    })
    .catch((err) => {
      checked = true;
      child.kill();
      console.error('❌ Smoke test failed:', err.message);
      process.exit(1);
    });
}, 800);

child.on('error', (err) => {
  console.error('❌ Failed to start server.js:', err.message);
  process.exit(1);
});

setTimeout(() => {
  if (!checked) {
    child.kill();
    console.error('❌ Server did not respond in time.');
    process.exit(1);
  }
}, 5000);