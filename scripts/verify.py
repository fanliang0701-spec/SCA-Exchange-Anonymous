"""Verify archive integrity, paper numbers, and pricing scenarios."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'analysis'))
from reanalyze import cost, first_at, joint_envelope, self_check, stable_at, winner


def check_archives():
    counts = {}
    for name, expected in [('rank-curves.zip',1490), ('residual-results.zip',380),
                           ('bitwise-rl-results.zip',30), ('second-block-results.zip',1640)]:
        with zipfile.ZipFile(PACKAGE / 'data' / name) as archive:
            manifest = json.loads(archive.read('manifest.json')) if name == 'rank-curves.zip' else None
            names = [n for n in archive.namelist() if n.endswith('.npy')]
            assert len(names) == expected, (name, len(names))
            for member in names:
                payload = archive.read(member)
                meta = manifest[member[:-4]] if manifest is not None else json.loads(archive.read(member[:-4]+'.json'))
                assert hashlib.sha256(payload).hexdigest() == meta['rank_npy_sha256'], member
                ranks = np.load(io.BytesIO(payload), allow_pickle=False)
                assert ranks.shape == (20,10000) and ranks.dtype == np.int16
                assert ranks.min() >= 0 and ranks.max() <= 510
            counts[name] = len(names)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PACKAGE/'results')
    args = parser.parse_args()
    self_check()
    counts = check_archives()
    result = json.loads((args.output/'extended-analysis.json').read_text(encoding='utf-8'))
    groups = result['groups']
    byid = {g['id']:g for g in groups}
    assert len(groups) == len(byid) == 190
    assert len(result['second_block']['groups']) == 164
    seeds = [13,37,73,101,137,211,251,307,401,503]
    assert all(g['seeds']==seeds and len(g['C_subsets'])==10 for g in groups)
    # All numeric entries of Table II (None denotes censored; missing denotes n/a).
    boards = ['D1','D2','D3','D4','AES_GPU']
    expected = {
        2000:{'HW-template-fixed':[None,1162,3929,1806,None], 'HW-RL':[None,599,940,1116,None],
              'Bit-Gauss':[None,2959,None,4448,None], 'Bitwise-RL':['missing']*4+[None],
              'Real':[None,3937,3521,4445,None], 'RShare-T':[None,277,394,455,None],
              'FM4-6000':[None,337,403,345,4974], 'DDPM16-6000':[9955,336,403,404,None]},
        5000:{'HW-template-fixed':[1738,844,949,780,2596], 'HW-RL':[980,404,358,379,1576],
              'Bit-Gauss':[2501,1175,2030,1269,376], 'Bitwise-RL':['missing']*4+[297],
              'Real':[5856,989,492,745,9592], 'RShare-T':[445,170,144,141,1585],
              'FM4-6000':[773,230,159,189,'missing'], 'DDPM16-6000':[875,240,155,193,'missing']}}
    for q, methods in expected.items():
        for method, values in methods.items():
            for b, value in zip(boards,values):
                item = byid.get(f'{b}/Q{q}/{method}')
                assert (item['A95'] if item else 'missing') == value, (b,q,method)
    cheap=[]; calibrated=[]
    for b in boards:
        subset=[g for g in groups if g['dataset']==b]
        cheap.append(winner(subset,.001,.001))
        calibrated.append(winner(subset,1/10.87,1/10.87))
        # Check exact envelopes against an independent brute-force finite grid.
        envelope=joint_envelope(subset)
        for price in np.geomspace(1e-7,1,101):
            segment=next(x for x in envelope if x['lo']<=price and (x['hi'] is None or price<x['hi']))
            assert abs(cost(byid[segment['id']],price,price)-winner(subset,price,price)['total'])<1e-7
    assert [g['method'] for g in cheap] == ['HW-RL']*4+['Bit-Gauss']
    assert [g['method'] for g in calibrated] == ['RShare-T']*4+['Bitwise-RL']
    assert [round(g['total'],2) for g in cheap] == [4.31,.73,1.07,1.25,3.50]
    assert [round(g['total'],2) for g in calibrated] == [325.23,33.40,46.20,48.66,308.52]
    assert round(result['bootstrap_crossings']['D1']['point']*1000,2)==17.00
    assert round(result['bootstrap_crossings']['AES_GPU']['point']*1000,2)==15.25
    second={g['id']:g for g in result['second_block']['groups']}
    assert second['D1/Q2000/DDPM16-6000']['A95'] is None
    assert second['D1/Q5000/HW-RL']['A95']==798
    assert result['scope']['capture_rate']['traces_per_second']==10.87
    for row in result['fidelity']:
        assert row['capture_reported']==winner([byid[k] for k in row['allowed']],1/10.87,1/10.87)
    report={'passed':True,'rank_arrays_verified':counts,'primary_groups':190,'second_block_groups':164,
            'table_II_cells_verified':80,'paper_costs_verified':True,'finite_grid_envelopes_verified':True,
            'equal_1ms':cheap,'calibrated_92ms':calibrated,
            'capture_reference_traces_per_second':10.87,
            'training_rerun':False,'physical_capture_repeated':False}
    (args.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('PASS: 3,540 rank arrays; all 80 Table II cells; five cost decisions; break-even and second-block results.')


if __name__=='__main__':main()
