from tests import TestCaseNeSVoR
import torch
import scipy
from scipy.sparse.linalg import cg as cg_scipy
from nesvor.svr.reconstruction import cg


class TestCG(TestCaseNeSVoR):
    def test_cg(self):
        A = torch.tensor(
            scipy.linalg.hankel([1, 2, 3, 4, 5], [4, 7, 7, 8, 9]), dtype=torch.float32
        ).cuda()
        n = n_iter = A.shape[0]
        b = torch.arange(n, dtype=A.dtype, device=A.device).reshape(-1, 1)
        x0 = torch.zeros_like(b)
        funcA = lambda x: A @ x
        x_ = cg(funcA, b, x0, n_iter)
        x, _ = cg_scipy(A.cpu().numpy(), b.cpu().numpy(), tol=0, maxiter=n_iter, atol=0)
        x = torch.tensor(x, dtype=x_.dtype, device=x_.device).reshape(x_.shape)
        self.assert_tensor_close(x_, x)
