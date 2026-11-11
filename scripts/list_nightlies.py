"""List every dated OpenTitan nightly DV report in the public reports.opentitan.org bucket."""
import json, re, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
API = "https://storage.googleapis.com/storage/v1/b/reports.opentitan.org/o?"
def lst(prefix):
    out, tok = [], None
    while True:
        q = {'prefix': prefix, 'delimiter': '/', 'maxResults': 1000}
        if tok: q['pageToken'] = tok
        d = json.load(urllib.request.urlopen(API + urllib.parse.urlencode(q), timeout=60))
        out += d.get('prefixes', []); tok = d.get('nextPageToken')
        if not tok: return out
ips = [p.split('/')[2] for p in lst('hw/ip/')]
def dates(ip):
    res = []
    for sub in lst(f'hw/ip/{ip}/dv/'):
        name = sub.rstrip('/').split('/')[-1]
        if re.match(r'\d{4}\.\d\d\.\d\d_\d\d\.\d\d\.\d\d$', name): res.append(sub)
        elif name == 'uvm':  # otbn keeps reports under dv/uvm/
            res += [s for s in lst(sub) if re.search(r'\d{4}\.\d\d\.\d\d_', s)]
    return ip, res
with ThreadPoolExecutor(8) as ex:
    m = dict(ex.map(dates, ips))
m = {k: v for k, v in m.items() if v}
json.dump(m, open('data/raw_index.json', 'w'), indent=0)
print(len(m), sum(len(v) for v in m.values()))
for k, v in sorted(m.items()): print(k, len(v))
