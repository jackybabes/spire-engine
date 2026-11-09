"""Where a soak run's game and engine first part ways in a fight, from its playback log.

    python3 tools/trace/soak_diff.py harness/soak/<dir> harness/soak/<dir>/logs/<log>...

The soak itself only compares floor and HP before each decision. This replays each logged run in
the engine (`sts-cli serve`, decisions from <dir>/batch.json) and compares what the log has from
the game at every combat step: the hand's order (a `play N` picks a card by position, so a
different hand plays a different card), the player's block, and each living monster's HP and
block. It prints the first step that differs, with both sides. Build sts-cli first.
"""
import json, re, subprocess, sys
soak = sys.argv[1]
b = json.load(open(f'{soak}/batch.json'))['runs']
PNAME = {"Weakened": "Weakened", "Vulnerable": "Vulnerable"}
def game_steps(path):
    steps, cur = {}, None
    for line in open(path):
        line = line[9:].rstrip('\n')
        m = re.match(r'step (\d+) ', line)
        if m:
            cur = int(m.group(1)); steps[cur] = {"mon": [], "player": None, "hand": None}; continue
        if cur is None: continue
        if line.startswith('   game monster '):
            mm = re.match(r'   game monster (.+?) (-?\d+) block (-?\d+) (.*)$', line)
            if mm:
                steps[cur]["mon"].append((mm.group(1), int(mm.group(2)), int(mm.group(3)), ' '.join(sorted(x for x in mm.group(4).split() if ':' in x))))
        elif line.startswith('   game player block'):
            parts = line[len('   game player block '):].split()
            steps[cur]["player"] = (int(parts[0]), ' '.join(sorted(parts[1:])))
        elif line.startswith('   game hand'):
            steps[cur]["hand"] = line
            steps[cur]["cards"] = [x.rstrip('+').split('+')[0] for x in eval(line[line.index('['):line.index(']') + 1])]
    return steps
for path in sys.argv[2:]:
    seed = path.split('-')[1] if '/' not in path else path.rsplit('/', 1)[1].split('-')[1]
    log = next(r for r in b if r['log']['seed_string'] == seed)['log']
    gs = game_steps(path)
    if not gs: continue
    last = max(gs)
    cmds = [{"cmd": "new_run", "seed": log['seed'], "ascension": log['ascension'], "neow_full": True}] + [{"cmd": "step", "action": s['action']} for s in log['steps'][:last]]
    out = [json.loads(l) for l in subprocess.run(['./target/release/sts-cli', 'serve'], input='\n'.join(map(json.dumps, cmds)) + '\n', capture_output=True, text=True).stdout.splitlines()]
    found = None
    for k in sorted(gs):
        g = gs[k]
        if not g["mon"] or g["player"] is None: continue
        c = out[k]['obs']['screen'].get('combat')
        if not c: continue
        alive = [m for m in c['monsters']]
        em = [(m.get('id') or m.get('name'), m.get('hp'), m.get('block'),
               ' '.join(sorted(f"{x['id']}:{x['amount']}" for x in m.get('powers', [])))) for m in alive]
        gm = [x for x in g["mon"]]
        ep = (c['player']['block'], ' '.join(sorted(f"{x['id']}:{x['amount']}" for x in c['player']['powers'])))
        ghp = [(x[1], x[2]) for x in gm if x[1] > 0]
        ehp = [(x[1], x[2]) for x in em if x[1] > 0]
        eh = [h['name'].split('+')[0] for h in c['hand']]
        gh = g.get("cards")
        if gh is not None and gh != eh:
            print(f"   hand order differs before decision {k}: game {gh} engine {eh}; previous action {log['steps'][k-1]['action']}")
            found = (k, g, gm, em, ep)
            break
        if ghp != ehp or g["player"][0] != ep[0]:
            found = (k, g, gm, em, ep); break
    print(f"=== {path.rsplit('/',1)[-1]}")
    if not found:
        print("   no monster HP/block or player block difference in logged steps"); continue
    k, g, gm, em, ep = found
    print(f"   first difference before decision {k}; previous action: {log['steps'][k-1]['action']}")
    print(f"   game   player block {g['player'][0]} {g['player'][1]}")
    print(f"   engine player block {ep[0]} {ep[1]}")
    for x in gm: print(f"   game   {x}")
    for x in em: print(f"   engine {x}")
    prev = gs.get(k - 1)
    if prev and prev["hand"]: print("   game before:", prev["hand"].strip(), [x[:3] for x in prev["mon"]])
