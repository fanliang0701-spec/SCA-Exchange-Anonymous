"""AES lookup and recorded Gaussian likelihood scoring."""
import torch

def sbox(device):
    data=bytes.fromhex('637c777bf26b6fc53001672bfed7ab76ca82c97dfa5947f0add4a2af9ca472c0'
      'b7fd9326363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b275'
      '09832c1a1b6e5aa0523bd6b329e32f8453d100ed20fcb15b6acbbe394a4c58cf'
      'd0efaafb434d338545f9027f503c9fa851a3408f929d38f5bcb6da2110fff3d2'
      'cd0c13ec5f974417c4a77e3d645d197360814fdc222a908846eeb814de5e0bdb'
      'e0323a0a4906245cc2d3ac629195e479e7c8376d8dd54ea96c56f4ea657aae08'
      'ba78252e1ca6b4c6e8dd741f4bbd8b8a703eb5664803f60e613557b986c11d9e'
      'e1f8981169d98e949b1e87e9ce5528df8ca1890dbfe6426841992d0fb054bb16')
    return torch.tensor(list(data),device=device,dtype=torch.long)


def gaussian_scores(x,means,cov):
    # Candidate-independent -x^T C^-1 x/2 and log determinant cancel.
    precision_means=torch.linalg.solve(cov,means.T)
    return x@precision_means-.5*(means*precision_means.T).sum(1)

