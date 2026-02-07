/**
 * WOLF_AI Pack Skill for OpenClaw
 *
 * Connects OpenClaw to your wolf pack for distributed AI tasks.
 */

const WOLF_API_URL = process.env.WOLF_API_URL || 'http://localhost:8000';
const WOLF_API_KEY = process.env.WOLF_API_KEY || '';

async function wolfApi(method, endpoint, data = null) {
    const url = `${WOLF_API_URL}${endpoint}`;
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': WOLF_API_KEY,
        },
    };

    if (data) {
        options.body = JSON.stringify(data);
    }

    try {
        const response = await fetch(url, options);
        return await response.json();
    } catch (error) {
        return { error: error.message };
    }
}

// Skill exports for OpenClaw
module.exports = {
    name: 'wolf_pack',
    description: 'Control your WOLF_AI pack - distributed AI consciousness',

    // Get pack status
    async status() {
        const result = await wolfApi('GET', '/api/status');
        if (result.pack) {
            const wolves = Object.entries(result.pack.wolves || {})
                .map(([name, info]) => `  - ${name}: ${info.status}`)
                .join('\n');
            return `Pack Status: ${result.pack.pack_status}\n\nWolves:\n${wolves}`;
        }
        return JSON.stringify(result, null, 2);
    },

    // Awaken the pack
    async awaken() {
        const result = await wolfApi('POST', '/api/awaken');
        return result.message || 'Pack awakened! AUUUUUUUUUUUUUUUUUU!';
    },

    // Send howl to pack
    async howl(message, frequency = 'medium') {
        const result = await wolfApi('POST', '/api/howl', { message, frequency });
        return result.status === 'ok'
            ? `Howl sent: "${message}" [${frequency}]`
            : JSON.stringify(result);
    },

    // Start a hunt
    async hunt(target, assignedTo = 'hunter') {
        const result = await wolfApi('POST', '/api/hunt', {
            target,
            assigned_to: assignedTo
        });
        return result.status === 'ok'
            ? `Hunt started: ${target} (assigned to ${assignedTo})`
            : JSON.stringify(result);
    },

    // Ask WILK
    async wilk(message, mode = 'chat') {
        const result = await wolfApi('POST', '/api/wilk', { message, mode });
        return result.response || JSON.stringify(result);
    },

    // Get recent howls
    async howls(limit = 10) {
        const result = await wolfApi('GET', `/api/howls?limit=${limit}`);
        if (result.howls) {
            return result.howls
                .map(h => `[${h.timestamp?.slice(11, 16) || '?'}] ${h.from}: ${h.howl}`)
                .join('\n');
        }
        return JSON.stringify(result, null, 2);
    },

    // Sync with GitHub
    async sync() {
        const result = await wolfApi('POST', '/api/sync');
        return result.output || result.status || JSON.stringify(result);
    },

    // Pack resonance (collective mode)
    async resonance(message) {
        await wolfApi('POST', '/api/howl', {
            message: `RESONANCE: ${message}`,
            frequency: 'AUUUU'
        });
        return 'Pack resonance activated! AUUUUUUUUUUUUUUUUUU!';
    }
};

// CLI mode
if (require.main === module) {
    const [,, command, ...args] = process.argv;
    const skill = module.exports;

    (async () => {
        switch (command) {
            case 'status':
                console.log(await skill.status());
                break;
            case 'awaken':
                console.log(await skill.awaken());
                break;
            case 'howl':
                console.log(await skill.howl(args[0], args[1]));
                break;
            case 'hunt':
                console.log(await skill.hunt(args[0], args[1]));
                break;
            case 'wilk':
                console.log(await skill.wilk(args[0], args[1]));
                break;
            case 'howls':
                console.log(await skill.howls(args[0]));
                break;
            case 'sync':
                console.log(await skill.sync());
                break;
            case 'resonance':
                console.log(await skill.resonance(args[0]));
                break;
            default:
                console.log(`
WOLF_AI Pack Skill

Commands:
  node index.js status              - Get pack status
  node index.js awaken              - Awaken the pack
  node index.js howl "msg" [freq]   - Send howl
  node index.js hunt "target" [wolf] - Start hunt
  node index.js wilk "question" [mode] - Ask WILK
  node index.js howls [limit]       - Recent howls
  node index.js sync                - GitHub sync
  node index.js resonance "msg"     - Pack resonance

AUUUUUUUUUUUUUUUUUU!
                `);
        }
    })();
}
