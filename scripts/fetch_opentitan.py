"""Fetch every dated OpenTitan nightly DV report (public, reports.opentitan.org) and
parse it into compact JSON: per-test runs/passes/max runtime and dvsim failure buckets
(signature -> test -> failing-run count). Raw HTML is not stored.

usage: python3 scripts/fetch_opentitan.py  (needs data/raw_index.json from list_nightlies.py)
"""
import html, json, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://reports.opentitan.org/"
RT = re.compile(r'^(\d+(?:\.\d+)?)(s|m|h)$')

def txt(s):
    s = re.sub(r'<br\s*/?>', ' ', s)
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', '', s))).strip()

def secs(v):
    m = RT.match(v); x = float(m.group(1))
    return x * {'s': 1, 'm': 60, 'h': 3600}[m.group(2)]

def parse(page):
    out = {}
    m = re.search(r'GitHub Revision:\s*(?:<[^>]+>)*\s*([0-9a-f]{7,40})', page)
    out['rev'] = m.group(1) if m else None
    m = re.search(r'Simulator:\s*([A-Za-z]+)', page)
    out['sim'] = m.group(1) if m else None
    tests = {}
    head, _, buckets_html = page.partition('Failure Buckets')
    for row in re.findall(r'<tr[^>]*>(.*?)</tr>', head, re.S):
        cells = [txt(c) for c in re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)]
        for i, c in enumerate(cells):
            if RT.match(c) and i >= 1 and i + 3 < len(cells) and re.match(r'^[A-Za-z0-9_]+$', cells[i - 1]):
                name = cells[i - 1]
                try: p, n = int(cells[i + 2]), int(cells[i + 3])
                except ValueError: break
                if name not in tests: tests[name] = {'runs': n, 'pass': p, 'rt': round(secs(c), 3)}
                break
    out['tests'] = tests
    buckets = []
    # top-level bucket items: <li><p ...><code>SIG</code> has N failures:</p> ... Test X has k failures.
    parts = re.split(r'<li>\s*<p[^>]*>\s*<code>', buckets_html)
    for part in parts[1:]:
        sig_html, _, rest = part.partition('</code>')
        m = re.match(r'\s*has (\d+) failures', txt(rest[:200]))
        if not m: continue
        per = {}
        for t, k in re.findall(r'Test (\S+) has (\d+) failures', txt(rest)):
            per[t] = per.get(t, 0) + int(k)
        seeds = re.findall(r'(\d+)\.([A-Za-z0-9_]+)\.(\d{6,})', txt(rest))
        buckets.append({'sig': txt(sig_html), 'n': int(m.group(1)), 'tests': per,
                        'witness': [f"{a}.{b}.{c}" for a, b, c in seeds[:6]]})
    out['buckets'] = buckets
    return out

def job(item):
    ip, prefix = item
    date = prefix.rstrip('/').split('/')[-1]
    try:
        page = urllib.request.urlopen(BASE + prefix + 'report.html', timeout=60).read().decode('utf-8', 'replace')
    except Exception as e:
        return ip, date, {'error': str(e)[:80]}
    try:
        r = parse(page)
    except Exception as e:
        r = {'error': 'parse: ' + str(e)[:80]}
    return ip, date, r

if __name__ == '__main__':
    idx = json.load(open('data/raw_index.json'))
    items = [(ip, p) for ip, ps in idx.items() for p in ps]
    if len(sys.argv) > 1: items = items[:int(sys.argv[1])]
    res = {}
    with ThreadPoolExecutor(16) as ex:
        for k, (ip, date, r) in enumerate(ex.map(job, items)):
            res.setdefault(ip, {})[date] = r
            if k % 400 == 0: print(k, ip, date, len(r.get('tests', {})), len(r.get('buckets', [])), flush=True)
    json.dump(res, open('data/opentitan_nightlies.json', 'w'), separators=(',', ':'), sort_keys=True)
    errs = sum(1 for ip in res for d in res[ip] if 'error' in res[ip][d])
    print('done', sum(len(v) for v in res.values()), 'errors', errs)
