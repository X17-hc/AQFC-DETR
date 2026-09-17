"""Control-plane comparison names and loss invariants, without executing CUDA."""
from tools.benchmark_control import schedule, summarize_spatial
from tools.benchmark_interleaved import parser
import pytest


def test_explicit_backend_order():
    args=parser().parse_args(['--data-root','d','--pretrained','w','--output-dir','o',
        '--comparison','spatial','--order','reference,optimized,optimized,reference','--workers','2'])
    assert args.comparison=='spatial' and not args.profile and args.workers==2
    assert len(schedule(args.order,2,variants=('reference','optimized')))==4
    with pytest.raises(ValueError): schedule('p1,p2',2,variants=('reference','optimized'))


def test_spatial_summary_is_not_classification_comparison():
    rows=[dict(variant=v,images_per_second=s) for v,s in
          [('reference',2),('optimized',2.2),('optimized',2.2),('reference',2)]]
    result=summarize_spatial(rows)
    assert result['optimized_throughput_gain_percent']==pytest.approx(10)
    assert result['paired_gain_percent']==pytest.approx([10,10])
    assert 'p2_throughput_penalty_percent' not in result
