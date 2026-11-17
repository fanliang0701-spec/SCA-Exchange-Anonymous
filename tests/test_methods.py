"""Small numerical checks; these do not reproduce benchmark recovery rates."""
import tempfile
import unittest
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import likelihood, gaussian, classifier, base, residual_numpy, residual, aes, ranking, reference_ranks, generators, generator_training, cnn, hw_selection, smote


class MethodChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(13)
        cls.device=torch.device('cpu')
        cls.x=torch.randn(80,32)
        cls.y=torch.arange(80)%256

    def test_exact_kde_and_bit_basis(self):
        self.assertTrue(all(likelihood.check_numerics(self.device).values()))

    def test_frozen_hw_parameter_equivalence(self):
        old=gaussian.fit_gaussian(self.x,self.y,'HW',.01)
        new=likelihood.fit_mean(self.x,self.y)
        w=likelihood.whiten(new,.1)
        torch.testing.assert_close(old['means'],new['means'],rtol=1e-9,atol=1e-9)
        torch.testing.assert_close(old['cov'],w['L']@w['L'].T,rtol=1e-9,atol=1e-9)
        reps=[(1<<h)-1 for h in range(9)]
        old_scores=aes.gaussian_scores(self.x.double(),old['means'][reps],old['cov'])
        new_scores=likelihood.scores_gaussian(self.x,w)
        # The residual implementation omits the class-independent density constant.
        torch.testing.assert_close(old_scores-old_scores[:,:1],new_scores-new_scores[:,:1],rtol=1e-9,atol=1e-9)

    def test_residual_donors_and_prefix(self):
        st=gaussian.fit_gaussian(self.x,self.y,'HW',.01)
        targets=self.y.repeat(2)
        actual,donors,_=residual.transplant(st['means'],self.x,self.y,targets,torch.Generator().manual_seed(81))
        bank=residual_numpy.center_residuals(self.x.numpy(),st['means'][self.y].numpy())
        expected=residual_numpy.transplant(st['means'][targets].numpy(),bank,donors.numpy())
        np.testing.assert_allclose(actual.numpy(),expected,rtol=1e-6,atol=1e-7)
        pools,labels=base.make_pools(self.x,self.y,200,21)
        for px,py in pools.values():
            self.assertTrue(torch.equal(px[:80],self.x) and torch.equal(py[:80],self.y))
        counts=torch.bincount(labels)[self.y.unique()]
        self.assertLessEqual(int(counts.max()-counts.min()),1)

    def test_rank_ties_and_independent_prefix(self):
        rng=np.random.default_rng(42)
        scores=rng.normal(size=(133,256));scores[:,1]=scores[:,0]
        orders=np.stack([rng.permutation(133) for _ in range(3)])
        rank2=(2*ranking.rank_curves(scores,0,orders,self.device)).astype(np.int16)
        np.testing.assert_array_equal(rank2,reference_ranks.block_rank2(scores,0,orders,self.device))
        self.assertTrue(np.all(rank2%2==1))

    def test_generator_fixed_endpoints(self):
        for kind,steps in [('fm',4),('ddpm',16)]:
            model=generators.ConditionalGenerator(kind)
            if kind=='ddpm':model.clip_min=self.x.amin(0);model.clip_max=self.x.amax(0)
            with tempfile.TemporaryDirectory() as tmp:
                info=generator_training.train_measured(generators,model,self.x,self.y,self.x[:16],self.y[:16],13,tmp,steps=2,endpoints=(2,))
                model.load_state_dict(torch.load(Path(tmp)/'step2.pt',weights_only=True))
                out=generators.sample(model,kind,self.x[:7],self.y[:7],steps)
                self.assertEqual(out.shape,(7,32));self.assertTrue(torch.isfinite(out).all())
                self.assertEqual(model.sampling_info['nfe'],steps)
                self.assertIn('2',info['endpoint_fit_seconds'])

    def test_priors_selection_and_cnn(self):
        _,prior=classifier.priors(self.y)
        self.assertAlmostEqual(float(prior.sum()),1.)
        st,choice=hw_selection.tune_cv(hw_selection.common,self.x,self.y)
        self.assertEqual(choice['validation_rows_used'],0)
        self.assertEqual(choice['folds'],5)
        self.assertEqual(cnn.build_zaid(13,self.device)(self.x[:4]).shape,(4,9))
        self.assertEqual(classifier.make_model('hw9',13,self.device)(self.x[:4]).shape,(4,9))
        generated,labels,info=smote.make_smote(self.x,self.y,self.y[:5],13)
        self.assertTrue(torch.equal(generated,self.x[:5]))
        self.assertEqual(info['metadata']['singleton_generated_count'],5)


if __name__=='__main__':unittest.main()
