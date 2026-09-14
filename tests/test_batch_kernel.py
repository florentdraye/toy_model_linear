import unittest
import numpy as np
import torch
from src.config import ModelConfig
from src.model import ToyTransformer
from src.batch_kernel import reference_directions, projected_contributions, validate_aggregate, count_vector, sample_batches


class BatchKernelTest(unittest.TestCase):
    def test_products_against_explicit_jacobians(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = ToyTransformer(ModelConfig(vocab_size=3, seq_len=3, n_classes=4,
                    d_model=4, n_heads=1, d_ff=8, n_blocks=1, mlp_activation='relu2')).double().eval()
        x = torch.tensor([[0,1,2],[1,2,0],[2,0,1]])
        y = torch.tensor([1,2,3])
        directions, meta = reference_directions(model, x[:1], y[0])
        actual = projected_contributions(model, x, y, directions, 2)
        params = tuple(model.parameters())
        def jacobian(edge):
            z = model(edge)[0]
            return torch.stack([torch.cat([g.flatten() for g in torch.autograd.grad(v, params, retain_graph=True)]) for v in z])
        ji = jacobian(x[:1]); gi = torch.tensor(meta['g'], dtype=torch.float64)
        for j in range(3):
            jj = jacobian(x[j:j+1])
            gj = model(x[j:j+1])[0].softmax(-1).detach();gj[y[j]] -= 1
            vector = ji @ jj.T @ gj
            torch.testing.assert_close(actual[j],torch.stack([vector[y[0]],gi@vector]),rtol=1e-9,atol=1e-9)
        validate_aggregate(model,x[:1],y[0],x,y,actual)

    def test_exact_counts_and_identity_independent_control(self):
        classes=[0,1,2];p=[.2,.3,.5]
        for n in [0,1,16,32]:
            c=count_vector(classes,p,1,n,32)
            self.assertEqual(c.sum(),32);self.assertEqual(c[1],n)
        labels=np.repeat(classes,4)
        values=np.repeat([2.,5.,-1.],4)[None,:,None]
        a=sample_batches(values,labels,classes,p,1,[0,8,32],32,12)
        expected=a['class_counts']@np.array([2.,5.,-1.])
        np.testing.assert_allclose(a['responses'][:,:,0,0],np.repeat(expected[:,None],12,axis=1))
        np.testing.assert_array_equal(a['predicted_sd'],0)
        np.testing.assert_array_equal(a['responses'],sample_batches(values,labels,classes,p,1,[0,8,32],32,12)['responses'])
