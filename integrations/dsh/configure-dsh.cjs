/* Configure only DSH's local DeepSeek route; never reads stored credentials. */
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { createRequire } = require('node:module');
const { pathToFileURL } = require('node:url');

async function main() {
  const argv = process.argv.slice(2);
  const apply = argv.includes('--apply');
  const homeFlag = argv.indexOf('--home');
  const dshHome = path.resolve(homeFlag >= 0 ? argv[homeFlag + 1] : (process.env.DSH_HOME || path.join(os.homedir(), '.dsh')));
  const packageRoot = process.env.DSH_PACKAGE_ROOT || path.join(process.env.APPDATA, 'npm', 'node_modules', '@deepseek-ai', 'dsh');
  const requireDsh = createRequire(path.join(packageRoot, 'package.json'));
  const { parseDocument } = requireDsh('yaml');
  const { resolveAdapterOptions } = await import(pathToFileURL(requireDsh.resolve('@deepseek-ai/dsh-llm-deepseek')));
  const { withFileLock, writeFileAtomic } = await import(pathToFileURL(requireDsh.resolve('@deepseek-ai/dsh-atomic-write')));
  const filename = path.join(dshHome, 'settings.yaml');
  const model = 'deepseek-v4.1-flash-local';
  const route = {
    baseURL: 'http://127.0.0.1:48241/v1',
    apiKeyEnv: 'DSV41_LOCAL_API_KEY',
    reasoningEffort: 'high',
    maxTokens: 262144,
    defaultContextWindow: 1048576,
    streamIdleTimeoutMs: 300000,
    models: [{
      id: model,
      name: 'DeepSeek V4.1 Flash - ModelScope MI300X',
      description: 'Self-hosted Core v2 + ExpertPack + Engram; text and tool calling.',
      contextWindow: 1048576,
      maxTokens: 262144,
      inputModalities: ['text'],
    }],
  };
  resolveAdapterOptions(route);
  async function build() {
    let before = '';
    try { before = await fs.readFile(filename, 'utf8'); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    const doc = parseDocument(before || '{}\n');
    if (doc.errors.length) throw new Error('Existing settings YAML is invalid; no changes made.');
    const current = doc.toJSON();
    if (!current || typeof current !== 'object' || Array.isArray(current)) throw new Error('Settings must be a mapping.');
    for (const [key, value] of Object.entries(route)) doc.setIn(['llm-deepseek', key], value);
    doc.setIn(['agent-default-model', 'provider'], 'deepseek-official');
    doc.setIn(['agent-default-model', 'model'], model);
    doc.setIn(['agent-default-model', 'reasoningEffort'], 'high');
    const after = doc.toString();
    const parsed = parseDocument(after).toJSON();
    for (const key of Object.keys(current)) {
      if (!['llm-deepseek', 'agent-default-model'].includes(key) && JSON.stringify(current[key]) !== JSON.stringify(parsed[key])) {
        throw new Error(`Unrelated section would change: ${key}`);
      }
    }
    resolveAdapterOptions(parsed['llm-deepseek']);
    return { before, after };
  }
  if (!apply) {
    await build();
    console.log(JSON.stringify({ validated: true, applied: false, settingsPath: filename, route }, null, 2));
    return;
  }
  await fs.mkdir(dshHome, { recursive: true });
  await withFileLock(filename, async () => {
    const { before, after } = await build();
    if (before === after) { console.log('DSH_CONFIG_ALREADY_CURRENT'); return; }
    const backups = path.join(dshHome, 'backups');
    await fs.mkdir(backups, { recursive: true });
    const backup = path.join(backups, 'settings-before-dsv41-' + new Date().toISOString().replace(/[:.]/g, '-') + '.yaml');
    await fs.writeFile(backup, before, { flag: 'wx', mode: 0o600 });
    await writeFileAtomic(filename, after, { mode: 0o600, dirMode: 0o700 });
    console.log('DSH_CONFIG_APPLIED', filename);
    console.log('DSH_CONFIG_BACKUP', backup);
  });
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
